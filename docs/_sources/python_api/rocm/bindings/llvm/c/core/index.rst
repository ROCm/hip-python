rocm.bindings.llvm.c.core
=========================

.. py:module:: rocm.bindings.llvm.c.core


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.core.LLVMAttributeReturnIndex
   rocm.bindings.llvm.c.core.LLVMAttributeFunctionIndex
   rocm.bindings.llvm.c.core.LLVMFastMathAllowReassoc
   rocm.bindings.llvm.c.core.LLVMFastMathNoNaNs
   rocm.bindings.llvm.c.core.LLVMFastMathNoInfs
   rocm.bindings.llvm.c.core.LLVMFastMathNoSignedZeros
   rocm.bindings.llvm.c.core.LLVMFastMathAllowReciprocal
   rocm.bindings.llvm.c.core.LLVMFastMathAllowContract
   rocm.bindings.llvm.c.core.LLVMFastMathApproxFunc
   rocm.bindings.llvm.c.core.LLVMFastMathNone
   rocm.bindings.llvm.c.core.LLVMFastMathAll
   rocm.bindings.llvm.c.core.LLVMGEPFlagInBounds
   rocm.bindings.llvm.c.core.LLVMGEPFlagNUSW
   rocm.bindings.llvm.c.core.LLVMGEPFlagNUW


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.core.LLVMOpcode
   rocm.bindings.llvm.c.core.LLVMTypeKind
   rocm.bindings.llvm.c.core.LLVMLinkage
   rocm.bindings.llvm.c.core.LLVMVisibility
   rocm.bindings.llvm.c.core.LLVMUnnamedAddr
   rocm.bindings.llvm.c.core.LLVMDLLStorageClass
   rocm.bindings.llvm.c.core.LLVMCallConv
   rocm.bindings.llvm.c.core.LLVMValueKind
   rocm.bindings.llvm.c.core.LLVMIntPredicate
   rocm.bindings.llvm.c.core.LLVMRealPredicate
   rocm.bindings.llvm.c.core.LLVMThreadLocalMode
   rocm.bindings.llvm.c.core.LLVMAtomicOrdering
   rocm.bindings.llvm.c.core.LLVMAtomicRMWBinOp
   rocm.bindings.llvm.c.core.LLVMDiagnosticSeverity
   rocm.bindings.llvm.c.core.LLVMInlineAsmDialect
   rocm.bindings.llvm.c.core.LLVMModuleFlagBehavior
   rocm.bindings.llvm.c.core.LLVMTailCallKind
   rocm.bindings.llvm.c.core.LLVMDbgRecordKind
   rocm.bindings.llvm.c.core.LLVMDiagnosticHandler
   rocm.bindings.llvm.c.core.LLVMYieldCallback
   rocm.bindings.llvm.c.core.LLVMDenormalModeKind


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.core.has_symbol
   rocm.bindings.llvm.c.core.LLVMShutdown
   rocm.bindings.llvm.c.core.LLVMGetVersion
   rocm.bindings.llvm.c.core.LLVMCreateMessage
   rocm.bindings.llvm.c.core.LLVMDisposeMessage
   rocm.bindings.llvm.c.core.LLVMContextCreate
   rocm.bindings.llvm.c.core.LLVMGetGlobalContext
   rocm.bindings.llvm.c.core.LLVMContextSetDiagnosticHandler
   rocm.bindings.llvm.c.core.LLVMContextGetDiagnosticHandler
   rocm.bindings.llvm.c.core.LLVMContextGetDiagnosticContext
   rocm.bindings.llvm.c.core.LLVMContextSetYieldCallback
   rocm.bindings.llvm.c.core.LLVMContextShouldDiscardValueNames
   rocm.bindings.llvm.c.core.LLVMContextSetDiscardValueNames
   rocm.bindings.llvm.c.core.LLVMContextDispose
   rocm.bindings.llvm.c.core.LLVMGetDiagInfoDescription
   rocm.bindings.llvm.c.core.LLVMGetDiagInfoSeverity
   rocm.bindings.llvm.c.core.LLVMGetMDKindIDInContext
   rocm.bindings.llvm.c.core.LLVMGetMDKindID
   rocm.bindings.llvm.c.core.LLVMGetSyncScopeID
   rocm.bindings.llvm.c.core.LLVMGetEnumAttributeKindForName
   rocm.bindings.llvm.c.core.LLVMGetLastEnumAttributeKind
   rocm.bindings.llvm.c.core.LLVMCreateEnumAttribute
   rocm.bindings.llvm.c.core.LLVMGetEnumAttributeKind
   rocm.bindings.llvm.c.core.LLVMGetEnumAttributeValue
   rocm.bindings.llvm.c.core.LLVMCreateTypeAttribute
   rocm.bindings.llvm.c.core.LLVMGetTypeAttributeValue
   rocm.bindings.llvm.c.core.LLVMCreateConstantRangeAttribute
   rocm.bindings.llvm.c.core.LLVMCreateDenormalFPEnvAttribute
   rocm.bindings.llvm.c.core.LLVMCreateStringAttribute
   rocm.bindings.llvm.c.core.LLVMGetStringAttributeKind
   rocm.bindings.llvm.c.core.LLVMGetStringAttributeValue
   rocm.bindings.llvm.c.core.LLVMIsEnumAttribute
   rocm.bindings.llvm.c.core.LLVMIsStringAttribute
   rocm.bindings.llvm.c.core.LLVMIsTypeAttribute
   rocm.bindings.llvm.c.core.LLVMGetTypeByName2
   rocm.bindings.llvm.c.core.LLVMModuleCreateWithName
   rocm.bindings.llvm.c.core.LLVMModuleCreateWithNameInContext
   rocm.bindings.llvm.c.core.LLVMCloneModule
   rocm.bindings.llvm.c.core.LLVMDisposeModule
   rocm.bindings.llvm.c.core.LLVMIsNewDbgInfoFormat
   rocm.bindings.llvm.c.core.LLVMSetIsNewDbgInfoFormat
   rocm.bindings.llvm.c.core.LLVMGetModuleIdentifier
   rocm.bindings.llvm.c.core.LLVMSetModuleIdentifier
   rocm.bindings.llvm.c.core.LLVMGetSourceFileName
   rocm.bindings.llvm.c.core.LLVMSetSourceFileName
   rocm.bindings.llvm.c.core.LLVMGetDataLayoutStr
   rocm.bindings.llvm.c.core.LLVMGetDataLayout
   rocm.bindings.llvm.c.core.LLVMSetDataLayout
   rocm.bindings.llvm.c.core.LLVMGetTarget
   rocm.bindings.llvm.c.core.LLVMSetTarget
   rocm.bindings.llvm.c.core.LLVMCopyModuleFlagsMetadata
   rocm.bindings.llvm.c.core.LLVMDisposeModuleFlagsMetadata
   rocm.bindings.llvm.c.core.LLVMModuleFlagEntriesGetFlagBehavior
   rocm.bindings.llvm.c.core.LLVMModuleFlagEntriesGetKey
   rocm.bindings.llvm.c.core.LLVMModuleFlagEntriesGetMetadata
   rocm.bindings.llvm.c.core.LLVMGetModuleFlag
   rocm.bindings.llvm.c.core.LLVMAddModuleFlag
   rocm.bindings.llvm.c.core.LLVMDumpModule
   rocm.bindings.llvm.c.core.LLVMPrintModuleToFile
   rocm.bindings.llvm.c.core.LLVMPrintModuleToString
   rocm.bindings.llvm.c.core.LLVMGetModuleInlineAsm
   rocm.bindings.llvm.c.core.LLVMSetModuleInlineAsm2
   rocm.bindings.llvm.c.core.LLVMAppendModuleInlineAsm
   rocm.bindings.llvm.c.core.LLVMGetInlineAsm
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmAsmString
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmConstraintString
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmDialect
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmFunctionType
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmHasSideEffects
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmNeedsAlignedStack
   rocm.bindings.llvm.c.core.LLVMGetInlineAsmCanUnwind
   rocm.bindings.llvm.c.core.LLVMGetModuleContext
   rocm.bindings.llvm.c.core.LLVMGetTypeByName
   rocm.bindings.llvm.c.core.LLVMGetFirstNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetLastNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetNextNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetPreviousNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetOrInsertNamedMetadata
   rocm.bindings.llvm.c.core.LLVMGetNamedMetadataName
   rocm.bindings.llvm.c.core.LLVMGetNamedMetadataNumOperands
   rocm.bindings.llvm.c.core.LLVMGetNamedMetadataOperands
   rocm.bindings.llvm.c.core.LLVMAddNamedMetadataOperand
   rocm.bindings.llvm.c.core.LLVMGetDebugLocDirectory
   rocm.bindings.llvm.c.core.LLVMGetDebugLocFilename
   rocm.bindings.llvm.c.core.LLVMGetDebugLocLine
   rocm.bindings.llvm.c.core.LLVMGetDebugLocColumn
   rocm.bindings.llvm.c.core.LLVMAddFunction
   rocm.bindings.llvm.c.core.LLVMGetOrInsertFunction
   rocm.bindings.llvm.c.core.LLVMGetNamedFunction
   rocm.bindings.llvm.c.core.LLVMGetNamedFunctionWithLength
   rocm.bindings.llvm.c.core.LLVMGetFirstFunction
   rocm.bindings.llvm.c.core.LLVMGetLastFunction
   rocm.bindings.llvm.c.core.LLVMGetNextFunction
   rocm.bindings.llvm.c.core.LLVMGetPreviousFunction
   rocm.bindings.llvm.c.core.LLVMSetModuleInlineAsm
   rocm.bindings.llvm.c.core.LLVMGetTypeKind
   rocm.bindings.llvm.c.core.LLVMTypeIsSized
   rocm.bindings.llvm.c.core.LLVMGetTypeContext
   rocm.bindings.llvm.c.core.LLVMDumpType
   rocm.bindings.llvm.c.core.LLVMPrintTypeToString
   rocm.bindings.llvm.c.core.LLVMByteTypeInContext
   rocm.bindings.llvm.c.core.LLVMGetByteTypeWidth
   rocm.bindings.llvm.c.core.LLVMInt1TypeInContext
   rocm.bindings.llvm.c.core.LLVMInt8TypeInContext
   rocm.bindings.llvm.c.core.LLVMInt16TypeInContext
   rocm.bindings.llvm.c.core.LLVMInt32TypeInContext
   rocm.bindings.llvm.c.core.LLVMInt64TypeInContext
   rocm.bindings.llvm.c.core.LLVMInt128TypeInContext
   rocm.bindings.llvm.c.core.LLVMIntTypeInContext
   rocm.bindings.llvm.c.core.LLVMInt1Type
   rocm.bindings.llvm.c.core.LLVMInt8Type
   rocm.bindings.llvm.c.core.LLVMInt16Type
   rocm.bindings.llvm.c.core.LLVMInt32Type
   rocm.bindings.llvm.c.core.LLVMInt64Type
   rocm.bindings.llvm.c.core.LLVMInt128Type
   rocm.bindings.llvm.c.core.LLVMIntType
   rocm.bindings.llvm.c.core.LLVMGetIntTypeWidth
   rocm.bindings.llvm.c.core.LLVMHalfTypeInContext
   rocm.bindings.llvm.c.core.LLVMBFloatTypeInContext
   rocm.bindings.llvm.c.core.LLVMFloatTypeInContext
   rocm.bindings.llvm.c.core.LLVMDoubleTypeInContext
   rocm.bindings.llvm.c.core.LLVMX86FP80TypeInContext
   rocm.bindings.llvm.c.core.LLVMFP128TypeInContext
   rocm.bindings.llvm.c.core.LLVMPPCFP128TypeInContext
   rocm.bindings.llvm.c.core.LLVMHalfType
   rocm.bindings.llvm.c.core.LLVMBFloatType
   rocm.bindings.llvm.c.core.LLVMFloatType
   rocm.bindings.llvm.c.core.LLVMDoubleType
   rocm.bindings.llvm.c.core.LLVMX86FP80Type
   rocm.bindings.llvm.c.core.LLVMFP128Type
   rocm.bindings.llvm.c.core.LLVMPPCFP128Type
   rocm.bindings.llvm.c.core.LLVMFunctionType
   rocm.bindings.llvm.c.core.LLVMIsFunctionVarArg
   rocm.bindings.llvm.c.core.LLVMGetReturnType
   rocm.bindings.llvm.c.core.LLVMCountParamTypes
   rocm.bindings.llvm.c.core.LLVMGetParamTypes
   rocm.bindings.llvm.c.core.LLVMStructTypeInContext
   rocm.bindings.llvm.c.core.LLVMStructType
   rocm.bindings.llvm.c.core.LLVMStructCreateNamed
   rocm.bindings.llvm.c.core.LLVMGetStructName
   rocm.bindings.llvm.c.core.LLVMStructSetBody
   rocm.bindings.llvm.c.core.LLVMCountStructElementTypes
   rocm.bindings.llvm.c.core.LLVMGetStructElementTypes
   rocm.bindings.llvm.c.core.LLVMStructGetTypeAtIndex
   rocm.bindings.llvm.c.core.LLVMIsPackedStruct
   rocm.bindings.llvm.c.core.LLVMIsOpaqueStruct
   rocm.bindings.llvm.c.core.LLVMIsLiteralStruct
   rocm.bindings.llvm.c.core.LLVMGetElementType
   rocm.bindings.llvm.c.core.LLVMGetSubtypes
   rocm.bindings.llvm.c.core.LLVMGetNumContainedTypes
   rocm.bindings.llvm.c.core.LLVMArrayType
   rocm.bindings.llvm.c.core.LLVMArrayType2
   rocm.bindings.llvm.c.core.LLVMGetArrayLength
   rocm.bindings.llvm.c.core.LLVMGetArrayLength2
   rocm.bindings.llvm.c.core.LLVMPointerType
   rocm.bindings.llvm.c.core.LLVMPointerTypeIsOpaque
   rocm.bindings.llvm.c.core.LLVMPointerTypeInContext
   rocm.bindings.llvm.c.core.LLVMGetPointerAddressSpace
   rocm.bindings.llvm.c.core.LLVMVectorType
   rocm.bindings.llvm.c.core.LLVMScalableVectorType
   rocm.bindings.llvm.c.core.LLVMGetVectorSize
   rocm.bindings.llvm.c.core.LLVMGetConstantPtrAuthPointer
   rocm.bindings.llvm.c.core.LLVMGetConstantPtrAuthKey
   rocm.bindings.llvm.c.core.LLVMGetConstantPtrAuthDiscriminator
   rocm.bindings.llvm.c.core.LLVMGetConstantPtrAuthAddrDiscriminator
   rocm.bindings.llvm.c.core.LLVMVoidTypeInContext
   rocm.bindings.llvm.c.core.LLVMLabelTypeInContext
   rocm.bindings.llvm.c.core.LLVMX86AMXTypeInContext
   rocm.bindings.llvm.c.core.LLVMTokenTypeInContext
   rocm.bindings.llvm.c.core.LLVMMetadataTypeInContext
   rocm.bindings.llvm.c.core.LLVMVoidType
   rocm.bindings.llvm.c.core.LLVMLabelType
   rocm.bindings.llvm.c.core.LLVMX86AMXType
   rocm.bindings.llvm.c.core.LLVMTargetExtTypeInContext
   rocm.bindings.llvm.c.core.LLVMGetTargetExtTypeName
   rocm.bindings.llvm.c.core.LLVMGetTargetExtTypeNumTypeParams
   rocm.bindings.llvm.c.core.LLVMGetTargetExtTypeTypeParam
   rocm.bindings.llvm.c.core.LLVMGetTargetExtTypeNumIntParams
   rocm.bindings.llvm.c.core.LLVMGetTargetExtTypeIntParam
   rocm.bindings.llvm.c.core.LLVMTypeOf
   rocm.bindings.llvm.c.core.LLVMGetValueKind
   rocm.bindings.llvm.c.core.LLVMGetValueName2
   rocm.bindings.llvm.c.core.LLVMSetValueName2
   rocm.bindings.llvm.c.core.LLVMDumpValue
   rocm.bindings.llvm.c.core.LLVMPrintValueToString
   rocm.bindings.llvm.c.core.LLVMGetValueContext
   rocm.bindings.llvm.c.core.LLVMPrintDbgRecordToString
   rocm.bindings.llvm.c.core.LLVMReplaceAllUsesWith
   rocm.bindings.llvm.c.core.LLVMIsConstant
   rocm.bindings.llvm.c.core.LLVMIsUndef
   rocm.bindings.llvm.c.core.LLVMIsPoison
   rocm.bindings.llvm.c.core.LLVMIsAArgument
   rocm.bindings.llvm.c.core.LLVMIsABasicBlock
   rocm.bindings.llvm.c.core.LLVMIsAInlineAsm
   rocm.bindings.llvm.c.core.LLVMIsAUser
   rocm.bindings.llvm.c.core.LLVMIsAConstant
   rocm.bindings.llvm.c.core.LLVMIsABlockAddress
   rocm.bindings.llvm.c.core.LLVMIsAConstantAggregateZero
   rocm.bindings.llvm.c.core.LLVMIsAConstantArray
   rocm.bindings.llvm.c.core.LLVMIsAConstantDataSequential
   rocm.bindings.llvm.c.core.LLVMIsAConstantDataArray
   rocm.bindings.llvm.c.core.LLVMIsAConstantDataVector
   rocm.bindings.llvm.c.core.LLVMIsAConstantExpr
   rocm.bindings.llvm.c.core.LLVMIsAConstantFP
   rocm.bindings.llvm.c.core.LLVMIsAConstantInt
   rocm.bindings.llvm.c.core.LLVMIsAConstantByte
   rocm.bindings.llvm.c.core.LLVMIsAConstantPointerNull
   rocm.bindings.llvm.c.core.LLVMIsAConstantStruct
   rocm.bindings.llvm.c.core.LLVMIsAConstantTokenNone
   rocm.bindings.llvm.c.core.LLVMIsAConstantVector
   rocm.bindings.llvm.c.core.LLVMIsAConstantPtrAuth
   rocm.bindings.llvm.c.core.LLVMIsAGlobalValue
   rocm.bindings.llvm.c.core.LLVMIsAGlobalAlias
   rocm.bindings.llvm.c.core.LLVMIsAGlobalObject
   rocm.bindings.llvm.c.core.LLVMIsAFunction
   rocm.bindings.llvm.c.core.LLVMIsAGlobalVariable
   rocm.bindings.llvm.c.core.LLVMIsAGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMIsAUndefValue
   rocm.bindings.llvm.c.core.LLVMIsAPoisonValue
   rocm.bindings.llvm.c.core.LLVMIsAInstruction
   rocm.bindings.llvm.c.core.LLVMIsAUnaryOperator
   rocm.bindings.llvm.c.core.LLVMIsABinaryOperator
   rocm.bindings.llvm.c.core.LLVMIsACallInst
   rocm.bindings.llvm.c.core.LLVMIsAIntrinsicInst
   rocm.bindings.llvm.c.core.LLVMIsADbgInfoIntrinsic
   rocm.bindings.llvm.c.core.LLVMIsADbgVariableIntrinsic
   rocm.bindings.llvm.c.core.LLVMIsADbgDeclareInst
   rocm.bindings.llvm.c.core.LLVMIsADbgLabelInst
   rocm.bindings.llvm.c.core.LLVMIsAMemIntrinsic
   rocm.bindings.llvm.c.core.LLVMIsAMemCpyInst
   rocm.bindings.llvm.c.core.LLVMIsAMemMoveInst
   rocm.bindings.llvm.c.core.LLVMIsAMemSetInst
   rocm.bindings.llvm.c.core.LLVMIsACmpInst
   rocm.bindings.llvm.c.core.LLVMIsAFCmpInst
   rocm.bindings.llvm.c.core.LLVMIsAICmpInst
   rocm.bindings.llvm.c.core.LLVMIsAExtractElementInst
   rocm.bindings.llvm.c.core.LLVMIsAGetElementPtrInst
   rocm.bindings.llvm.c.core.LLVMIsAInsertElementInst
   rocm.bindings.llvm.c.core.LLVMIsAInsertValueInst
   rocm.bindings.llvm.c.core.LLVMIsALandingPadInst
   rocm.bindings.llvm.c.core.LLVMIsAPHINode
   rocm.bindings.llvm.c.core.LLVMIsASelectInst
   rocm.bindings.llvm.c.core.LLVMIsAShuffleVectorInst
   rocm.bindings.llvm.c.core.LLVMIsAStoreInst
   rocm.bindings.llvm.c.core.LLVMIsAUncondBrInst
   rocm.bindings.llvm.c.core.LLVMIsACondBrInst
   rocm.bindings.llvm.c.core.LLVMIsAIndirectBrInst
   rocm.bindings.llvm.c.core.LLVMIsAInvokeInst
   rocm.bindings.llvm.c.core.LLVMIsAReturnInst
   rocm.bindings.llvm.c.core.LLVMIsASwitchInst
   rocm.bindings.llvm.c.core.LLVMIsAUnreachableInst
   rocm.bindings.llvm.c.core.LLVMIsAResumeInst
   rocm.bindings.llvm.c.core.LLVMIsACleanupReturnInst
   rocm.bindings.llvm.c.core.LLVMIsACatchReturnInst
   rocm.bindings.llvm.c.core.LLVMIsACatchSwitchInst
   rocm.bindings.llvm.c.core.LLVMIsACallBrInst
   rocm.bindings.llvm.c.core.LLVMIsAFuncletPadInst
   rocm.bindings.llvm.c.core.LLVMIsACatchPadInst
   rocm.bindings.llvm.c.core.LLVMIsACleanupPadInst
   rocm.bindings.llvm.c.core.LLVMIsAUnaryInstruction
   rocm.bindings.llvm.c.core.LLVMIsAAllocaInst
   rocm.bindings.llvm.c.core.LLVMIsACastInst
   rocm.bindings.llvm.c.core.LLVMIsAAddrSpaceCastInst
   rocm.bindings.llvm.c.core.LLVMIsABitCastInst
   rocm.bindings.llvm.c.core.LLVMIsAFPExtInst
   rocm.bindings.llvm.c.core.LLVMIsAFPToSIInst
   rocm.bindings.llvm.c.core.LLVMIsAFPToUIInst
   rocm.bindings.llvm.c.core.LLVMIsAFPTruncInst
   rocm.bindings.llvm.c.core.LLVMIsAIntToPtrInst
   rocm.bindings.llvm.c.core.LLVMIsAPtrToIntInst
   rocm.bindings.llvm.c.core.LLVMIsASExtInst
   rocm.bindings.llvm.c.core.LLVMIsASIToFPInst
   rocm.bindings.llvm.c.core.LLVMIsATruncInst
   rocm.bindings.llvm.c.core.LLVMIsAUIToFPInst
   rocm.bindings.llvm.c.core.LLVMIsAZExtInst
   rocm.bindings.llvm.c.core.LLVMIsAExtractValueInst
   rocm.bindings.llvm.c.core.LLVMIsALoadInst
   rocm.bindings.llvm.c.core.LLVMIsAVAArgInst
   rocm.bindings.llvm.c.core.LLVMIsAFreezeInst
   rocm.bindings.llvm.c.core.LLVMIsAAtomicCmpXchgInst
   rocm.bindings.llvm.c.core.LLVMIsAAtomicRMWInst
   rocm.bindings.llvm.c.core.LLVMIsAFenceInst
   rocm.bindings.llvm.c.core.LLVMIsABranchInst
   rocm.bindings.llvm.c.core.LLVMIsAMDNode
   rocm.bindings.llvm.c.core.LLVMIsAValueAsMetadata
   rocm.bindings.llvm.c.core.LLVMIsAMDString
   rocm.bindings.llvm.c.core.LLVMGetValueName
   rocm.bindings.llvm.c.core.LLVMSetValueName
   rocm.bindings.llvm.c.core.LLVMGetFirstUse
   rocm.bindings.llvm.c.core.LLVMGetNextUse
   rocm.bindings.llvm.c.core.LLVMGetUser
   rocm.bindings.llvm.c.core.LLVMGetUsedValue
   rocm.bindings.llvm.c.core.LLVMGetOperand
   rocm.bindings.llvm.c.core.LLVMGetOperandUse
   rocm.bindings.llvm.c.core.LLVMSetOperand
   rocm.bindings.llvm.c.core.LLVMGetNumOperands
   rocm.bindings.llvm.c.core.LLVMConstNull
   rocm.bindings.llvm.c.core.LLVMConstAllOnes
   rocm.bindings.llvm.c.core.LLVMGetUndef
   rocm.bindings.llvm.c.core.LLVMGetPoison
   rocm.bindings.llvm.c.core.LLVMIsNull
   rocm.bindings.llvm.c.core.LLVMConstPointerNull
   rocm.bindings.llvm.c.core.LLVMConstInt
   rocm.bindings.llvm.c.core.LLVMConstIntOfArbitraryPrecision
   rocm.bindings.llvm.c.core.LLVMConstIntOfString
   rocm.bindings.llvm.c.core.LLVMConstIntOfStringAndSize
   rocm.bindings.llvm.c.core.LLVMConstByte
   rocm.bindings.llvm.c.core.LLVMConstByteOfArbitraryPrecision
   rocm.bindings.llvm.c.core.LLVMConstByteOfStringAndSize
   rocm.bindings.llvm.c.core.LLVMConstReal
   rocm.bindings.llvm.c.core.LLVMConstRealOfString
   rocm.bindings.llvm.c.core.LLVMConstRealOfStringAndSize
   rocm.bindings.llvm.c.core.LLVMConstFPFromBits
   rocm.bindings.llvm.c.core.LLVMConstIntGetZExtValue
   rocm.bindings.llvm.c.core.LLVMConstIntGetSExtValue
   rocm.bindings.llvm.c.core.LLVMConstByteGetZExtValue
   rocm.bindings.llvm.c.core.LLVMConstByteGetSExtValue
   rocm.bindings.llvm.c.core.LLVMConstRealGetDouble
   rocm.bindings.llvm.c.core.LLVMConstStringInContext
   rocm.bindings.llvm.c.core.LLVMConstStringInContext2
   rocm.bindings.llvm.c.core.LLVMConstString
   rocm.bindings.llvm.c.core.LLVMIsConstantString
   rocm.bindings.llvm.c.core.LLVMGetAsString
   rocm.bindings.llvm.c.core.LLVMGetRawDataValues
   rocm.bindings.llvm.c.core.LLVMConstStructInContext
   rocm.bindings.llvm.c.core.LLVMConstStruct
   rocm.bindings.llvm.c.core.LLVMConstArray
   rocm.bindings.llvm.c.core.LLVMConstArray2
   rocm.bindings.llvm.c.core.LLVMConstDataArray
   rocm.bindings.llvm.c.core.LLVMConstNamedStruct
   rocm.bindings.llvm.c.core.LLVMGetAggregateElement
   rocm.bindings.llvm.c.core.LLVMGetElementAsConstant
   rocm.bindings.llvm.c.core.LLVMConstVector
   rocm.bindings.llvm.c.core.LLVMConstantPtrAuth
   rocm.bindings.llvm.c.core.LLVMGetConstOpcode
   rocm.bindings.llvm.c.core.LLVMAlignOf
   rocm.bindings.llvm.c.core.LLVMSizeOf
   rocm.bindings.llvm.c.core.LLVMConstNeg
   rocm.bindings.llvm.c.core.LLVMConstNSWNeg
   rocm.bindings.llvm.c.core.LLVMConstNUWNeg
   rocm.bindings.llvm.c.core.LLVMConstNot
   rocm.bindings.llvm.c.core.LLVMConstAdd
   rocm.bindings.llvm.c.core.LLVMConstNSWAdd
   rocm.bindings.llvm.c.core.LLVMConstNUWAdd
   rocm.bindings.llvm.c.core.LLVMConstSub
   rocm.bindings.llvm.c.core.LLVMConstNSWSub
   rocm.bindings.llvm.c.core.LLVMConstNUWSub
   rocm.bindings.llvm.c.core.LLVMConstXor
   rocm.bindings.llvm.c.core.LLVMConstGEP2
   rocm.bindings.llvm.c.core.LLVMConstInBoundsGEP2
   rocm.bindings.llvm.c.core.LLVMConstGEPWithNoWrapFlags
   rocm.bindings.llvm.c.core.LLVMConstTrunc
   rocm.bindings.llvm.c.core.LLVMConstPtrToInt
   rocm.bindings.llvm.c.core.LLVMConstIntToPtr
   rocm.bindings.llvm.c.core.LLVMConstBitCast
   rocm.bindings.llvm.c.core.LLVMConstAddrSpaceCast
   rocm.bindings.llvm.c.core.LLVMConstTruncOrBitCast
   rocm.bindings.llvm.c.core.LLVMConstPointerCast
   rocm.bindings.llvm.c.core.LLVMConstExtractElement
   rocm.bindings.llvm.c.core.LLVMConstInsertElement
   rocm.bindings.llvm.c.core.LLVMConstShuffleVector
   rocm.bindings.llvm.c.core.LLVMBlockAddress
   rocm.bindings.llvm.c.core.LLVMGetBlockAddressFunction
   rocm.bindings.llvm.c.core.LLVMGetBlockAddressBasicBlock
   rocm.bindings.llvm.c.core.LLVMConstInlineAsm
   rocm.bindings.llvm.c.core.LLVMGetGlobalParent
   rocm.bindings.llvm.c.core.LLVMIsDeclaration
   rocm.bindings.llvm.c.core.LLVMGetLinkage
   rocm.bindings.llvm.c.core.LLVMSetLinkage
   rocm.bindings.llvm.c.core.LLVMGetSection
   rocm.bindings.llvm.c.core.LLVMSetSection
   rocm.bindings.llvm.c.core.LLVMGetVisibility
   rocm.bindings.llvm.c.core.LLVMSetVisibility
   rocm.bindings.llvm.c.core.LLVMGetDLLStorageClass
   rocm.bindings.llvm.c.core.LLVMSetDLLStorageClass
   rocm.bindings.llvm.c.core.LLVMGetUnnamedAddress
   rocm.bindings.llvm.c.core.LLVMSetUnnamedAddress
   rocm.bindings.llvm.c.core.LLVMGlobalGetValueType
   rocm.bindings.llvm.c.core.LLVMHasUnnamedAddr
   rocm.bindings.llvm.c.core.LLVMSetUnnamedAddr
   rocm.bindings.llvm.c.core.LLVMGetAlignment
   rocm.bindings.llvm.c.core.LLVMSetAlignment
   rocm.bindings.llvm.c.core.LLVMGlobalSetMetadata
   rocm.bindings.llvm.c.core.LLVMGlobalAddMetadata
   rocm.bindings.llvm.c.core.LLVMGlobalEraseMetadata
   rocm.bindings.llvm.c.core.LLVMGlobalClearMetadata
   rocm.bindings.llvm.c.core.LLVMGlobalAddDebugInfo
   rocm.bindings.llvm.c.core.LLVMGlobalCopyAllMetadata
   rocm.bindings.llvm.c.core.LLVMDisposeValueMetadataEntries
   rocm.bindings.llvm.c.core.LLVMValueMetadataEntriesGetKind
   rocm.bindings.llvm.c.core.LLVMValueMetadataEntriesGetMetadata
   rocm.bindings.llvm.c.core.LLVMAddGlobal
   rocm.bindings.llvm.c.core.LLVMAddGlobalInAddressSpace
   rocm.bindings.llvm.c.core.LLVMGetNamedGlobal
   rocm.bindings.llvm.c.core.LLVMGetNamedGlobalWithLength
   rocm.bindings.llvm.c.core.LLVMGetFirstGlobal
   rocm.bindings.llvm.c.core.LLVMGetLastGlobal
   rocm.bindings.llvm.c.core.LLVMGetNextGlobal
   rocm.bindings.llvm.c.core.LLVMGetPreviousGlobal
   rocm.bindings.llvm.c.core.LLVMDeleteGlobal
   rocm.bindings.llvm.c.core.LLVMGetInitializer
   rocm.bindings.llvm.c.core.LLVMSetInitializer
   rocm.bindings.llvm.c.core.LLVMIsThreadLocal
   rocm.bindings.llvm.c.core.LLVMSetThreadLocal
   rocm.bindings.llvm.c.core.LLVMIsGlobalConstant
   rocm.bindings.llvm.c.core.LLVMSetGlobalConstant
   rocm.bindings.llvm.c.core.LLVMGetThreadLocalMode
   rocm.bindings.llvm.c.core.LLVMSetThreadLocalMode
   rocm.bindings.llvm.c.core.LLVMIsExternallyInitialized
   rocm.bindings.llvm.c.core.LLVMSetExternallyInitialized
   rocm.bindings.llvm.c.core.LLVMAddAlias2
   rocm.bindings.llvm.c.core.LLVMGetNamedGlobalAlias
   rocm.bindings.llvm.c.core.LLVMGetFirstGlobalAlias
   rocm.bindings.llvm.c.core.LLVMGetLastGlobalAlias
   rocm.bindings.llvm.c.core.LLVMGetNextGlobalAlias
   rocm.bindings.llvm.c.core.LLVMGetPreviousGlobalAlias
   rocm.bindings.llvm.c.core.LLVMAliasGetAliasee
   rocm.bindings.llvm.c.core.LLVMAliasSetAliasee
   rocm.bindings.llvm.c.core.LLVMDeleteFunction
   rocm.bindings.llvm.c.core.LLVMHasPersonalityFn
   rocm.bindings.llvm.c.core.LLVMGetPersonalityFn
   rocm.bindings.llvm.c.core.LLVMSetPersonalityFn
   rocm.bindings.llvm.c.core.LLVMLookupIntrinsicID
   rocm.bindings.llvm.c.core.LLVMGetIntrinsicID
   rocm.bindings.llvm.c.core.LLVMGetIntrinsicDeclaration
   rocm.bindings.llvm.c.core.LLVMIntrinsicGetType
   rocm.bindings.llvm.c.core.LLVMIntrinsicGetName
   rocm.bindings.llvm.c.core.LLVMIntrinsicCopyOverloadedName
   rocm.bindings.llvm.c.core.LLVMIntrinsicCopyOverloadedName2
   rocm.bindings.llvm.c.core.LLVMIntrinsicIsOverloaded
   rocm.bindings.llvm.c.core.LLVMGetFunctionCallConv
   rocm.bindings.llvm.c.core.LLVMSetFunctionCallConv
   rocm.bindings.llvm.c.core.LLVMGetGC
   rocm.bindings.llvm.c.core.LLVMSetGC
   rocm.bindings.llvm.c.core.LLVMGetPrefixData
   rocm.bindings.llvm.c.core.LLVMHasPrefixData
   rocm.bindings.llvm.c.core.LLVMSetPrefixData
   rocm.bindings.llvm.c.core.LLVMGetPrologueData
   rocm.bindings.llvm.c.core.LLVMHasPrologueData
   rocm.bindings.llvm.c.core.LLVMSetPrologueData
   rocm.bindings.llvm.c.core.LLVMAddAttributeAtIndex
   rocm.bindings.llvm.c.core.LLVMGetAttributeCountAtIndex
   rocm.bindings.llvm.c.core.LLVMGetAttributesAtIndex
   rocm.bindings.llvm.c.core.LLVMGetEnumAttributeAtIndex
   rocm.bindings.llvm.c.core.LLVMGetStringAttributeAtIndex
   rocm.bindings.llvm.c.core.LLVMRemoveEnumAttributeAtIndex
   rocm.bindings.llvm.c.core.LLVMRemoveStringAttributeAtIndex
   rocm.bindings.llvm.c.core.LLVMAddTargetDependentFunctionAttr
   rocm.bindings.llvm.c.core.LLVMCountParams
   rocm.bindings.llvm.c.core.LLVMGetParams
   rocm.bindings.llvm.c.core.LLVMGetParam
   rocm.bindings.llvm.c.core.LLVMGetParamParent
   rocm.bindings.llvm.c.core.LLVMGetFirstParam
   rocm.bindings.llvm.c.core.LLVMGetLastParam
   rocm.bindings.llvm.c.core.LLVMGetNextParam
   rocm.bindings.llvm.c.core.LLVMGetPreviousParam
   rocm.bindings.llvm.c.core.LLVMSetParamAlignment
   rocm.bindings.llvm.c.core.LLVMAddGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetNamedGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetFirstGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetLastGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetNextGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetPreviousGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMGetGlobalIFuncResolver
   rocm.bindings.llvm.c.core.LLVMSetGlobalIFuncResolver
   rocm.bindings.llvm.c.core.LLVMEraseGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMRemoveGlobalIFunc
   rocm.bindings.llvm.c.core.LLVMMDStringInContext2
   rocm.bindings.llvm.c.core.LLVMMDNodeInContext2
   rocm.bindings.llvm.c.core.LLVMMetadataAsValue
   rocm.bindings.llvm.c.core.LLVMValueAsMetadata
   rocm.bindings.llvm.c.core.LLVMGetMDString
   rocm.bindings.llvm.c.core.LLVMGetMDNodeNumOperands
   rocm.bindings.llvm.c.core.LLVMGetMDNodeOperands
   rocm.bindings.llvm.c.core.LLVMReplaceMDNodeOperandWith
   rocm.bindings.llvm.c.core.LLVMMDStringInContext
   rocm.bindings.llvm.c.core.LLVMMDString
   rocm.bindings.llvm.c.core.LLVMMDNodeInContext
   rocm.bindings.llvm.c.core.LLVMMDNode
   rocm.bindings.llvm.c.core.LLVMCreateOperandBundle
   rocm.bindings.llvm.c.core.LLVMDisposeOperandBundle
   rocm.bindings.llvm.c.core.LLVMGetOperandBundleTag
   rocm.bindings.llvm.c.core.LLVMGetNumOperandBundleArgs
   rocm.bindings.llvm.c.core.LLVMGetOperandBundleArgAtIndex
   rocm.bindings.llvm.c.core.LLVMBasicBlockAsValue
   rocm.bindings.llvm.c.core.LLVMValueIsBasicBlock
   rocm.bindings.llvm.c.core.LLVMValueAsBasicBlock
   rocm.bindings.llvm.c.core.LLVMGetBasicBlockName
   rocm.bindings.llvm.c.core.LLVMGetBasicBlockParent
   rocm.bindings.llvm.c.core.LLVMGetBasicBlockTerminator
   rocm.bindings.llvm.c.core.LLVMCountBasicBlocks
   rocm.bindings.llvm.c.core.LLVMGetBasicBlocks
   rocm.bindings.llvm.c.core.LLVMGetFirstBasicBlock
   rocm.bindings.llvm.c.core.LLVMGetLastBasicBlock
   rocm.bindings.llvm.c.core.LLVMGetNextBasicBlock
   rocm.bindings.llvm.c.core.LLVMGetPreviousBasicBlock
   rocm.bindings.llvm.c.core.LLVMGetEntryBasicBlock
   rocm.bindings.llvm.c.core.LLVMInsertExistingBasicBlockAfterInsertBlock
   rocm.bindings.llvm.c.core.LLVMAppendExistingBasicBlock
   rocm.bindings.llvm.c.core.LLVMCreateBasicBlockInContext
   rocm.bindings.llvm.c.core.LLVMAppendBasicBlockInContext
   rocm.bindings.llvm.c.core.LLVMAppendBasicBlock
   rocm.bindings.llvm.c.core.LLVMInsertBasicBlockInContext
   rocm.bindings.llvm.c.core.LLVMInsertBasicBlock
   rocm.bindings.llvm.c.core.LLVMDeleteBasicBlock
   rocm.bindings.llvm.c.core.LLVMRemoveBasicBlockFromParent
   rocm.bindings.llvm.c.core.LLVMMoveBasicBlockBefore
   rocm.bindings.llvm.c.core.LLVMMoveBasicBlockAfter
   rocm.bindings.llvm.c.core.LLVMGetFirstInstruction
   rocm.bindings.llvm.c.core.LLVMGetLastInstruction
   rocm.bindings.llvm.c.core.LLVMHasMetadata
   rocm.bindings.llvm.c.core.LLVMGetMetadata
   rocm.bindings.llvm.c.core.LLVMSetMetadata
   rocm.bindings.llvm.c.core.LLVMInstructionGetAllMetadataOtherThanDebugLoc
   rocm.bindings.llvm.c.core.LLVMGetInstructionParent
   rocm.bindings.llvm.c.core.LLVMGetNextInstruction
   rocm.bindings.llvm.c.core.LLVMGetPreviousInstruction
   rocm.bindings.llvm.c.core.LLVMInstructionRemoveFromParent
   rocm.bindings.llvm.c.core.LLVMInstructionEraseFromParent
   rocm.bindings.llvm.c.core.LLVMDeleteInstruction
   rocm.bindings.llvm.c.core.LLVMGetInstructionOpcode
   rocm.bindings.llvm.c.core.LLVMGetICmpPredicate
   rocm.bindings.llvm.c.core.LLVMGetICmpSameSign
   rocm.bindings.llvm.c.core.LLVMSetICmpSameSign
   rocm.bindings.llvm.c.core.LLVMGetFCmpPredicate
   rocm.bindings.llvm.c.core.LLVMInstructionClone
   rocm.bindings.llvm.c.core.LLVMIsATerminatorInst
   rocm.bindings.llvm.c.core.LLVMGetFirstDbgRecord
   rocm.bindings.llvm.c.core.LLVMGetLastDbgRecord
   rocm.bindings.llvm.c.core.LLVMGetNextDbgRecord
   rocm.bindings.llvm.c.core.LLVMGetPreviousDbgRecord
   rocm.bindings.llvm.c.core.LLVMDbgRecordGetDebugLoc
   rocm.bindings.llvm.c.core.LLVMDbgRecordGetKind
   rocm.bindings.llvm.c.core.LLVMDbgVariableRecordGetValue
   rocm.bindings.llvm.c.core.LLVMDbgVariableRecordGetVariable
   rocm.bindings.llvm.c.core.LLVMDbgVariableRecordGetExpression
   rocm.bindings.llvm.c.core.LLVMGetNumArgOperands
   rocm.bindings.llvm.c.core.LLVMSetInstructionCallConv
   rocm.bindings.llvm.c.core.LLVMGetInstructionCallConv
   rocm.bindings.llvm.c.core.LLVMSetInstrParamAlignment
   rocm.bindings.llvm.c.core.LLVMAddCallSiteAttribute
   rocm.bindings.llvm.c.core.LLVMGetCallSiteAttributeCount
   rocm.bindings.llvm.c.core.LLVMGetCallSiteAttributes
   rocm.bindings.llvm.c.core.LLVMGetCallSiteEnumAttribute
   rocm.bindings.llvm.c.core.LLVMGetCallSiteStringAttribute
   rocm.bindings.llvm.c.core.LLVMRemoveCallSiteEnumAttribute
   rocm.bindings.llvm.c.core.LLVMRemoveCallSiteStringAttribute
   rocm.bindings.llvm.c.core.LLVMGetCalledFunctionType
   rocm.bindings.llvm.c.core.LLVMGetCalledValue
   rocm.bindings.llvm.c.core.LLVMGetNumOperandBundles
   rocm.bindings.llvm.c.core.LLVMGetOperandBundleAtIndex
   rocm.bindings.llvm.c.core.LLVMIsTailCall
   rocm.bindings.llvm.c.core.LLVMSetTailCall
   rocm.bindings.llvm.c.core.LLVMGetTailCallKind
   rocm.bindings.llvm.c.core.LLVMSetTailCallKind
   rocm.bindings.llvm.c.core.LLVMGetNormalDest
   rocm.bindings.llvm.c.core.LLVMGetUnwindDest
   rocm.bindings.llvm.c.core.LLVMSetNormalDest
   rocm.bindings.llvm.c.core.LLVMSetUnwindDest
   rocm.bindings.llvm.c.core.LLVMGetCallBrDefaultDest
   rocm.bindings.llvm.c.core.LLVMGetCallBrNumIndirectDests
   rocm.bindings.llvm.c.core.LLVMGetCallBrIndirectDest
   rocm.bindings.llvm.c.core.LLVMGetNumSuccessors
   rocm.bindings.llvm.c.core.LLVMGetSuccessor
   rocm.bindings.llvm.c.core.LLVMSetSuccessor
   rocm.bindings.llvm.c.core.LLVMIsConditional
   rocm.bindings.llvm.c.core.LLVMGetCondition
   rocm.bindings.llvm.c.core.LLVMSetCondition
   rocm.bindings.llvm.c.core.LLVMGetSwitchDefaultDest
   rocm.bindings.llvm.c.core.LLVMGetSwitchCaseValue
   rocm.bindings.llvm.c.core.LLVMSetSwitchCaseValue
   rocm.bindings.llvm.c.core.LLVMGetAllocatedType
   rocm.bindings.llvm.c.core.LLVMIsInBounds
   rocm.bindings.llvm.c.core.LLVMSetIsInBounds
   rocm.bindings.llvm.c.core.LLVMGetGEPSourceElementType
   rocm.bindings.llvm.c.core.LLVMGEPGetNoWrapFlags
   rocm.bindings.llvm.c.core.LLVMGEPSetNoWrapFlags
   rocm.bindings.llvm.c.core.LLVMAddIncoming
   rocm.bindings.llvm.c.core.LLVMCountIncoming
   rocm.bindings.llvm.c.core.LLVMGetIncomingValue
   rocm.bindings.llvm.c.core.LLVMGetIncomingBlock
   rocm.bindings.llvm.c.core.LLVMGetNumIndices
   rocm.bindings.llvm.c.core.LLVMGetIndices
   rocm.bindings.llvm.c.core.LLVMCreateBuilderInContext
   rocm.bindings.llvm.c.core.LLVMCreateBuilder
   rocm.bindings.llvm.c.core.LLVMPositionBuilder
   rocm.bindings.llvm.c.core.LLVMPositionBuilderBeforeDbgRecords
   rocm.bindings.llvm.c.core.LLVMPositionBuilderBefore
   rocm.bindings.llvm.c.core.LLVMPositionBuilderBeforeInstrAndDbgRecords
   rocm.bindings.llvm.c.core.LLVMPositionBuilderAtEnd
   rocm.bindings.llvm.c.core.LLVMGetInsertBlock
   rocm.bindings.llvm.c.core.LLVMClearInsertionPosition
   rocm.bindings.llvm.c.core.LLVMInsertIntoBuilder
   rocm.bindings.llvm.c.core.LLVMInsertIntoBuilderWithName
   rocm.bindings.llvm.c.core.LLVMDisposeBuilder
   rocm.bindings.llvm.c.core.LLVMGetCurrentDebugLocation2
   rocm.bindings.llvm.c.core.LLVMSetCurrentDebugLocation2
   rocm.bindings.llvm.c.core.LLVMSetInstDebugLocation
   rocm.bindings.llvm.c.core.LLVMAddMetadataToInst
   rocm.bindings.llvm.c.core.LLVMBuilderGetDefaultFPMathTag
   rocm.bindings.llvm.c.core.LLVMBuilderSetDefaultFPMathTag
   rocm.bindings.llvm.c.core.LLVMGetBuilderContext
   rocm.bindings.llvm.c.core.LLVMSetCurrentDebugLocation
   rocm.bindings.llvm.c.core.LLVMGetCurrentDebugLocation
   rocm.bindings.llvm.c.core.LLVMBuildRetVoid
   rocm.bindings.llvm.c.core.LLVMBuildRet
   rocm.bindings.llvm.c.core.LLVMBuildAggregateRet
   rocm.bindings.llvm.c.core.LLVMBuildBr
   rocm.bindings.llvm.c.core.LLVMBuildCondBr
   rocm.bindings.llvm.c.core.LLVMBuildSwitch
   rocm.bindings.llvm.c.core.LLVMBuildIndirectBr
   rocm.bindings.llvm.c.core.LLVMBuildCallBr
   rocm.bindings.llvm.c.core.LLVMBuildInvoke2
   rocm.bindings.llvm.c.core.LLVMBuildInvokeWithOperandBundles
   rocm.bindings.llvm.c.core.LLVMBuildUnreachable
   rocm.bindings.llvm.c.core.LLVMBuildResume
   rocm.bindings.llvm.c.core.LLVMBuildLandingPad
   rocm.bindings.llvm.c.core.LLVMBuildCleanupRet
   rocm.bindings.llvm.c.core.LLVMBuildCatchRet
   rocm.bindings.llvm.c.core.LLVMBuildCatchPad
   rocm.bindings.llvm.c.core.LLVMBuildCleanupPad
   rocm.bindings.llvm.c.core.LLVMBuildCatchSwitch
   rocm.bindings.llvm.c.core.LLVMAddCase
   rocm.bindings.llvm.c.core.LLVMAddDestination
   rocm.bindings.llvm.c.core.LLVMGetNumClauses
   rocm.bindings.llvm.c.core.LLVMGetClause
   rocm.bindings.llvm.c.core.LLVMAddClause
   rocm.bindings.llvm.c.core.LLVMIsCleanup
   rocm.bindings.llvm.c.core.LLVMSetCleanup
   rocm.bindings.llvm.c.core.LLVMAddHandler
   rocm.bindings.llvm.c.core.LLVMGetNumHandlers
   rocm.bindings.llvm.c.core.LLVMGetHandlers
   rocm.bindings.llvm.c.core.LLVMGetArgOperand
   rocm.bindings.llvm.c.core.LLVMSetArgOperand
   rocm.bindings.llvm.c.core.LLVMGetParentCatchSwitch
   rocm.bindings.llvm.c.core.LLVMSetParentCatchSwitch
   rocm.bindings.llvm.c.core.LLVMBuildAdd
   rocm.bindings.llvm.c.core.LLVMBuildNSWAdd
   rocm.bindings.llvm.c.core.LLVMBuildNUWAdd
   rocm.bindings.llvm.c.core.LLVMBuildFAdd
   rocm.bindings.llvm.c.core.LLVMBuildSub
   rocm.bindings.llvm.c.core.LLVMBuildNSWSub
   rocm.bindings.llvm.c.core.LLVMBuildNUWSub
   rocm.bindings.llvm.c.core.LLVMBuildFSub
   rocm.bindings.llvm.c.core.LLVMBuildMul
   rocm.bindings.llvm.c.core.LLVMBuildNSWMul
   rocm.bindings.llvm.c.core.LLVMBuildNUWMul
   rocm.bindings.llvm.c.core.LLVMBuildFMul
   rocm.bindings.llvm.c.core.LLVMBuildUDiv
   rocm.bindings.llvm.c.core.LLVMBuildExactUDiv
   rocm.bindings.llvm.c.core.LLVMBuildSDiv
   rocm.bindings.llvm.c.core.LLVMBuildExactSDiv
   rocm.bindings.llvm.c.core.LLVMBuildFDiv
   rocm.bindings.llvm.c.core.LLVMBuildURem
   rocm.bindings.llvm.c.core.LLVMBuildSRem
   rocm.bindings.llvm.c.core.LLVMBuildFRem
   rocm.bindings.llvm.c.core.LLVMBuildShl
   rocm.bindings.llvm.c.core.LLVMBuildLShr
   rocm.bindings.llvm.c.core.LLVMBuildAShr
   rocm.bindings.llvm.c.core.LLVMBuildAnd
   rocm.bindings.llvm.c.core.LLVMBuildOr
   rocm.bindings.llvm.c.core.LLVMBuildXor
   rocm.bindings.llvm.c.core.LLVMBuildBinOp
   rocm.bindings.llvm.c.core.LLVMBuildNeg
   rocm.bindings.llvm.c.core.LLVMBuildNSWNeg
   rocm.bindings.llvm.c.core.LLVMBuildNUWNeg
   rocm.bindings.llvm.c.core.LLVMBuildFNeg
   rocm.bindings.llvm.c.core.LLVMBuildNot
   rocm.bindings.llvm.c.core.LLVMGetNUW
   rocm.bindings.llvm.c.core.LLVMSetNUW
   rocm.bindings.llvm.c.core.LLVMGetNSW
   rocm.bindings.llvm.c.core.LLVMSetNSW
   rocm.bindings.llvm.c.core.LLVMGetExact
   rocm.bindings.llvm.c.core.LLVMSetExact
   rocm.bindings.llvm.c.core.LLVMGetNNeg
   rocm.bindings.llvm.c.core.LLVMSetNNeg
   rocm.bindings.llvm.c.core.LLVMGetFastMathFlags
   rocm.bindings.llvm.c.core.LLVMSetFastMathFlags
   rocm.bindings.llvm.c.core.LLVMCanValueUseFastMathFlags
   rocm.bindings.llvm.c.core.LLVMGetIsDisjoint
   rocm.bindings.llvm.c.core.LLVMSetIsDisjoint
   rocm.bindings.llvm.c.core.LLVMBuildMalloc
   rocm.bindings.llvm.c.core.LLVMBuildArrayMalloc
   rocm.bindings.llvm.c.core.LLVMBuildMemSet
   rocm.bindings.llvm.c.core.LLVMBuildMemCpy
   rocm.bindings.llvm.c.core.LLVMBuildMemMove
   rocm.bindings.llvm.c.core.LLVMBuildAlloca
   rocm.bindings.llvm.c.core.LLVMBuildArrayAlloca
   rocm.bindings.llvm.c.core.LLVMBuildFree
   rocm.bindings.llvm.c.core.LLVMBuildLoad2
   rocm.bindings.llvm.c.core.LLVMBuildStore
   rocm.bindings.llvm.c.core.LLVMBuildGEP2
   rocm.bindings.llvm.c.core.LLVMBuildInBoundsGEP2
   rocm.bindings.llvm.c.core.LLVMBuildGEPWithNoWrapFlags
   rocm.bindings.llvm.c.core.LLVMBuildStructGEP2
   rocm.bindings.llvm.c.core.LLVMBuildGlobalString
   rocm.bindings.llvm.c.core.LLVMBuildGlobalStringPtr
   rocm.bindings.llvm.c.core.LLVMGetVolatile
   rocm.bindings.llvm.c.core.LLVMSetVolatile
   rocm.bindings.llvm.c.core.LLVMGetWeak
   rocm.bindings.llvm.c.core.LLVMSetWeak
   rocm.bindings.llvm.c.core.LLVMGetOrdering
   rocm.bindings.llvm.c.core.LLVMSetOrdering
   rocm.bindings.llvm.c.core.LLVMGetAtomicRMWBinOp
   rocm.bindings.llvm.c.core.LLVMSetAtomicRMWBinOp
   rocm.bindings.llvm.c.core.LLVMBuildTrunc
   rocm.bindings.llvm.c.core.LLVMBuildZExt
   rocm.bindings.llvm.c.core.LLVMBuildSExt
   rocm.bindings.llvm.c.core.LLVMBuildFPToUI
   rocm.bindings.llvm.c.core.LLVMBuildFPToSI
   rocm.bindings.llvm.c.core.LLVMBuildUIToFP
   rocm.bindings.llvm.c.core.LLVMBuildSIToFP
   rocm.bindings.llvm.c.core.LLVMBuildFPTrunc
   rocm.bindings.llvm.c.core.LLVMBuildFPExt
   rocm.bindings.llvm.c.core.LLVMBuildPtrToInt
   rocm.bindings.llvm.c.core.LLVMBuildIntToPtr
   rocm.bindings.llvm.c.core.LLVMBuildBitCast
   rocm.bindings.llvm.c.core.LLVMBuildAddrSpaceCast
   rocm.bindings.llvm.c.core.LLVMBuildZExtOrBitCast
   rocm.bindings.llvm.c.core.LLVMBuildSExtOrBitCast
   rocm.bindings.llvm.c.core.LLVMBuildTruncOrBitCast
   rocm.bindings.llvm.c.core.LLVMBuildCast
   rocm.bindings.llvm.c.core.LLVMBuildPointerCast
   rocm.bindings.llvm.c.core.LLVMBuildIntCast2
   rocm.bindings.llvm.c.core.LLVMBuildFPCast
   rocm.bindings.llvm.c.core.LLVMBuildIntCast
   rocm.bindings.llvm.c.core.LLVMGetCastOpcode
   rocm.bindings.llvm.c.core.LLVMBuildICmp
   rocm.bindings.llvm.c.core.LLVMBuildFCmp
   rocm.bindings.llvm.c.core.LLVMBuildPhi
   rocm.bindings.llvm.c.core.LLVMBuildCall2
   rocm.bindings.llvm.c.core.LLVMBuildCallWithOperandBundles
   rocm.bindings.llvm.c.core.LLVMBuildSelect
   rocm.bindings.llvm.c.core.LLVMBuildVAArg
   rocm.bindings.llvm.c.core.LLVMBuildExtractElement
   rocm.bindings.llvm.c.core.LLVMBuildInsertElement
   rocm.bindings.llvm.c.core.LLVMBuildShuffleVector
   rocm.bindings.llvm.c.core.LLVMBuildExtractValue
   rocm.bindings.llvm.c.core.LLVMBuildInsertValue
   rocm.bindings.llvm.c.core.LLVMBuildFreeze
   rocm.bindings.llvm.c.core.LLVMBuildIsNull
   rocm.bindings.llvm.c.core.LLVMBuildIsNotNull
   rocm.bindings.llvm.c.core.LLVMBuildPtrDiff2
   rocm.bindings.llvm.c.core.LLVMBuildFence
   rocm.bindings.llvm.c.core.LLVMBuildFenceSyncScope
   rocm.bindings.llvm.c.core.LLVMBuildAtomicRMW
   rocm.bindings.llvm.c.core.LLVMBuildAtomicRMWSyncScope
   rocm.bindings.llvm.c.core.LLVMBuildAtomicCmpXchg
   rocm.bindings.llvm.c.core.LLVMBuildAtomicCmpXchgSyncScope
   rocm.bindings.llvm.c.core.LLVMGetNumMaskElements
   rocm.bindings.llvm.c.core.LLVMGetUndefMaskElem
   rocm.bindings.llvm.c.core.LLVMGetMaskValue
   rocm.bindings.llvm.c.core.LLVMIsAtomicSingleThread
   rocm.bindings.llvm.c.core.LLVMSetAtomicSingleThread
   rocm.bindings.llvm.c.core.LLVMIsAtomic
   rocm.bindings.llvm.c.core.LLVMGetAtomicSyncScopeID
   rocm.bindings.llvm.c.core.LLVMSetAtomicSyncScopeID
   rocm.bindings.llvm.c.core.LLVMGetCmpXchgSuccessOrdering
   rocm.bindings.llvm.c.core.LLVMSetCmpXchgSuccessOrdering
   rocm.bindings.llvm.c.core.LLVMGetCmpXchgFailureOrdering
   rocm.bindings.llvm.c.core.LLVMSetCmpXchgFailureOrdering
   rocm.bindings.llvm.c.core.LLVMCreateModuleProviderForExistingModule
   rocm.bindings.llvm.c.core.LLVMDisposeModuleProvider
   rocm.bindings.llvm.c.core.LLVMCreateMemoryBufferWithContentsOfFile
   rocm.bindings.llvm.c.core.LLVMCreateMemoryBufferWithSTDIN
   rocm.bindings.llvm.c.core.LLVMCreateMemoryBufferWithMemoryRange
   rocm.bindings.llvm.c.core.LLVMCreateMemoryBufferWithMemoryRangeCopy
   rocm.bindings.llvm.c.core.LLVMGetBufferStart
   rocm.bindings.llvm.c.core.LLVMGetBufferSize
   rocm.bindings.llvm.c.core.LLVMDisposeMemoryBuffer
   rocm.bindings.llvm.c.core.LLVMCreatePassManager
   rocm.bindings.llvm.c.core.LLVMCreateFunctionPassManagerForModule
   rocm.bindings.llvm.c.core.LLVMCreateFunctionPassManager
   rocm.bindings.llvm.c.core.LLVMRunPassManager
   rocm.bindings.llvm.c.core.LLVMInitializeFunctionPassManager
   rocm.bindings.llvm.c.core.LLVMRunFunctionPassManager
   rocm.bindings.llvm.c.core.LLVMFinalizeFunctionPassManager
   rocm.bindings.llvm.c.core.LLVMDisposePassManager
   rocm.bindings.llvm.c.core.LLVMStartMultithreaded
   rocm.bindings.llvm.c.core.LLVMStopMultithreaded
   rocm.bindings.llvm.c.core.LLVMIsMultithreaded


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpcode

   Bases: :py:obj:`enum.IntEnum`


   External users depend on the following values being stable.

   It is not safe
   to reorder them.


   .. py:attribute:: LLVMRet
      :type:  int


   .. py:attribute:: LLVMUncondBr
      :type:  int


   .. py:attribute:: LLVMCondBr
      :type:  int


   .. py:attribute:: LLVMSwitch
      :type:  int


   .. py:attribute:: LLVMIndirectBr
      :type:  int


   .. py:attribute:: LLVMInvoke
      :type:  int


   .. py:attribute:: LLVMUnreachable
      :type:  int


   .. py:attribute:: LLVMCallBr
      :type:  int


   .. py:attribute:: LLVMFNeg
      :type:  int


   .. py:attribute:: LLVMAdd
      :type:  int


   .. py:attribute:: LLVMFAdd
      :type:  int


   .. py:attribute:: LLVMSub
      :type:  int


   .. py:attribute:: LLVMFSub
      :type:  int


   .. py:attribute:: LLVMMul
      :type:  int


   .. py:attribute:: LLVMFMul
      :type:  int


   .. py:attribute:: LLVMUDiv
      :type:  int


   .. py:attribute:: LLVMSDiv
      :type:  int


   .. py:attribute:: LLVMFDiv
      :type:  int


   .. py:attribute:: LLVMURem
      :type:  int


   .. py:attribute:: LLVMSRem
      :type:  int


   .. py:attribute:: LLVMFRem
      :type:  int


   .. py:attribute:: LLVMShl
      :type:  int


   .. py:attribute:: LLVMLShr
      :type:  int


   .. py:attribute:: LLVMAShr
      :type:  int


   .. py:attribute:: LLVMAnd
      :type:  int


   .. py:attribute:: LLVMOr
      :type:  int


   .. py:attribute:: LLVMXor
      :type:  int


   .. py:attribute:: LLVMAlloca
      :type:  int


   .. py:attribute:: LLVMLoad
      :type:  int


   .. py:attribute:: LLVMStore
      :type:  int


   .. py:attribute:: LLVMGetElementPtr
      :type:  int


   .. py:attribute:: LLVMTrunc
      :type:  int


   .. py:attribute:: LLVMZExt
      :type:  int


   .. py:attribute:: LLVMSExt
      :type:  int


   .. py:attribute:: LLVMFPToUI
      :type:  int


   .. py:attribute:: LLVMFPToSI
      :type:  int


   .. py:attribute:: LLVMUIToFP
      :type:  int


   .. py:attribute:: LLVMSIToFP
      :type:  int


   .. py:attribute:: LLVMFPTrunc
      :type:  int


   .. py:attribute:: LLVMFPExt
      :type:  int


   .. py:attribute:: LLVMPtrToInt
      :type:  int


   .. py:attribute:: LLVMPtrToAddr
      :type:  int


   .. py:attribute:: LLVMIntToPtr
      :type:  int


   .. py:attribute:: LLVMBitCast
      :type:  int


   .. py:attribute:: LLVMAddrSpaceCast
      :type:  int


   .. py:attribute:: LLVMICmp
      :type:  int


   .. py:attribute:: LLVMFCmp
      :type:  int


   .. py:attribute:: LLVMPHI
      :type:  int


   .. py:attribute:: LLVMCall
      :type:  int


   .. py:attribute:: LLVMSelect
      :type:  int


   .. py:attribute:: LLVMUserOp1
      :type:  int


   .. py:attribute:: LLVMUserOp2
      :type:  int


   .. py:attribute:: LLVMVAArg
      :type:  int


   .. py:attribute:: LLVMExtractElement
      :type:  int


   .. py:attribute:: LLVMInsertElement
      :type:  int


   .. py:attribute:: LLVMShuffleVector
      :type:  int


   .. py:attribute:: LLVMExtractValue
      :type:  int


   .. py:attribute:: LLVMInsertValue
      :type:  int


   .. py:attribute:: LLVMFreeze
      :type:  int


   .. py:attribute:: LLVMFence
      :type:  int


   .. py:attribute:: LLVMAtomicCmpXchg
      :type:  int


   .. py:attribute:: LLVMAtomicRMW
      :type:  int


   .. py:attribute:: LLVMResume
      :type:  int


   .. py:attribute:: LLVMLandingPad
      :type:  int


   .. py:attribute:: LLVMCleanupRet
      :type:  int


   .. py:attribute:: LLVMCatchRet
      :type:  int


   .. py:attribute:: LLVMCatchPad
      :type:  int


   .. py:attribute:: LLVMCleanupPad
      :type:  int


   .. py:attribute:: LLVMCatchSwitch
      :type:  int


.. py:class:: LLVMTypeKind

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMVoidTypeKind
      :type:  int


   .. py:attribute:: LLVMHalfTypeKind
      :type:  int


   .. py:attribute:: LLVMFloatTypeKind
      :type:  int


   .. py:attribute:: LLVMDoubleTypeKind
      :type:  int


   .. py:attribute:: LLVMX86_FP80TypeKind
      :type:  int


   .. py:attribute:: LLVMFP128TypeKind
      :type:  int


   .. py:attribute:: LLVMPPC_FP128TypeKind
      :type:  int


   .. py:attribute:: LLVMLabelTypeKind
      :type:  int


   .. py:attribute:: LLVMIntegerTypeKind
      :type:  int


   .. py:attribute:: LLVMFunctionTypeKind
      :type:  int


   .. py:attribute:: LLVMStructTypeKind
      :type:  int


   .. py:attribute:: LLVMArrayTypeKind
      :type:  int


   .. py:attribute:: LLVMPointerTypeKind
      :type:  int


   .. py:attribute:: LLVMVectorTypeKind
      :type:  int


   .. py:attribute:: LLVMMetadataTypeKind
      :type:  int


   .. py:attribute:: LLVMTokenTypeKind
      :type:  int


   .. py:attribute:: LLVMScalableVectorTypeKind
      :type:  int


   .. py:attribute:: LLVMBFloatTypeKind
      :type:  int


   .. py:attribute:: LLVMX86_AMXTypeKind
      :type:  int


   .. py:attribute:: LLVMTargetExtTypeKind
      :type:  int


   .. py:attribute:: LLVMByteTypeKind
      :type:  int


.. py:class:: LLVMLinkage

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMExternalLinkage
      :type:  int


   .. py:attribute:: LLVMAvailableExternallyLinkage
      :type:  int


   .. py:attribute:: LLVMLinkOnceAnyLinkage
      :type:  int


   .. py:attribute:: LLVMLinkOnceODRLinkage
      :type:  int


   .. py:attribute:: LLVMLinkOnceODRAutoHideLinkage
      :type:  int


   .. py:attribute:: LLVMWeakAnyLinkage
      :type:  int


   .. py:attribute:: LLVMWeakODRLinkage
      :type:  int


   .. py:attribute:: LLVMAppendingLinkage
      :type:  int


   .. py:attribute:: LLVMInternalLinkage
      :type:  int


   .. py:attribute:: LLVMPrivateLinkage
      :type:  int


   .. py:attribute:: LLVMDLLImportLinkage
      :type:  int


   .. py:attribute:: LLVMDLLExportLinkage
      :type:  int


   .. py:attribute:: LLVMExternalWeakLinkage
      :type:  int


   .. py:attribute:: LLVMGhostLinkage
      :type:  int


   .. py:attribute:: LLVMCommonLinkage
      :type:  int


   .. py:attribute:: LLVMLinkerPrivateLinkage
      :type:  int


   .. py:attribute:: LLVMLinkerPrivateWeakLinkage
      :type:  int


.. py:class:: LLVMVisibility

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMDefaultVisibility
      :type:  int


   .. py:attribute:: LLVMHiddenVisibility
      :type:  int


   .. py:attribute:: LLVMProtectedVisibility
      :type:  int


.. py:class:: LLVMUnnamedAddr

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMNoUnnamedAddr
      :type:  int


   .. py:attribute:: LLVMLocalUnnamedAddr
      :type:  int


   .. py:attribute:: LLVMGlobalUnnamedAddr
      :type:  int


.. py:class:: LLVMDLLStorageClass

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMDefaultStorageClass
      :type:  int


   .. py:attribute:: LLVMDLLImportStorageClass
      :type:  int


   .. py:attribute:: LLVMDLLExportStorageClass
      :type:  int


.. py:class:: LLVMCallConv

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMCCallConv
      :type:  int


   .. py:attribute:: LLVMFastCallConv
      :type:  int


   .. py:attribute:: LLVMColdCallConv
      :type:  int


   .. py:attribute:: LLVMGHCCallConv
      :type:  int


   .. py:attribute:: LLVMHiPECallConv
      :type:  int


   .. py:attribute:: LLVMAnyRegCallConv
      :type:  int


   .. py:attribute:: LLVMPreserveMostCallConv
      :type:  int


   .. py:attribute:: LLVMPreserveAllCallConv
      :type:  int


   .. py:attribute:: LLVMSwiftCallConv
      :type:  int


   .. py:attribute:: LLVMCXXFASTTLSCallConv
      :type:  int


   .. py:attribute:: LLVMX86StdcallCallConv
      :type:  int


   .. py:attribute:: LLVMX86FastcallCallConv
      :type:  int


   .. py:attribute:: LLVMARMAPCSCallConv
      :type:  int


   .. py:attribute:: LLVMARMAAPCSCallConv
      :type:  int


   .. py:attribute:: LLVMARMAAPCSVFPCallConv
      :type:  int


   .. py:attribute:: LLVMMSP430INTRCallConv
      :type:  int


   .. py:attribute:: LLVMX86ThisCallCallConv
      :type:  int


   .. py:attribute:: LLVMPTXKernelCallConv
      :type:  int


   .. py:attribute:: LLVMPTXDeviceCallConv
      :type:  int


   .. py:attribute:: LLVMSPIRFUNCCallConv
      :type:  int


   .. py:attribute:: LLVMSPIRKERNELCallConv
      :type:  int


   .. py:attribute:: LLVMIntelOCLBICallConv
      :type:  int


   .. py:attribute:: LLVMX8664SysVCallConv
      :type:  int


   .. py:attribute:: LLVMWin64CallConv
      :type:  int


   .. py:attribute:: LLVMX86VectorCallCallConv
      :type:  int


   .. py:attribute:: LLVMHHVMCallConv
      :type:  int


   .. py:attribute:: LLVMHHVMCCallConv
      :type:  int


   .. py:attribute:: LLVMX86INTRCallConv
      :type:  int


   .. py:attribute:: LLVMAVRINTRCallConv
      :type:  int


   .. py:attribute:: LLVMAVRSIGNALCallConv
      :type:  int


   .. py:attribute:: LLVMAVRBUILTINCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUVSCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUGSCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUPSCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUCSCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUKERNELCallConv
      :type:  int


   .. py:attribute:: LLVMX86RegCallCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUHSCallConv
      :type:  int


   .. py:attribute:: LLVMMSP430BUILTINCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPULSCallConv
      :type:  int


   .. py:attribute:: LLVMAMDGPUESCallConv
      :type:  int


.. py:class:: LLVMValueKind

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMArgumentValueKind
      :type:  int


   .. py:attribute:: LLVMBasicBlockValueKind
      :type:  int


   .. py:attribute:: LLVMMemoryUseValueKind
      :type:  int


   .. py:attribute:: LLVMMemoryDefValueKind
      :type:  int


   .. py:attribute:: LLVMMemoryPhiValueKind
      :type:  int


   .. py:attribute:: LLVMFunctionValueKind
      :type:  int


   .. py:attribute:: LLVMGlobalAliasValueKind
      :type:  int


   .. py:attribute:: LLVMGlobalIFuncValueKind
      :type:  int


   .. py:attribute:: LLVMGlobalVariableValueKind
      :type:  int


   .. py:attribute:: LLVMBlockAddressValueKind
      :type:  int


   .. py:attribute:: LLVMConstantExprValueKind
      :type:  int


   .. py:attribute:: LLVMConstantArrayValueKind
      :type:  int


   .. py:attribute:: LLVMConstantStructValueKind
      :type:  int


   .. py:attribute:: LLVMConstantVectorValueKind
      :type:  int


   .. py:attribute:: LLVMUndefValueValueKind
      :type:  int


   .. py:attribute:: LLVMConstantAggregateZeroValueKind
      :type:  int


   .. py:attribute:: LLVMConstantDataArrayValueKind
      :type:  int


   .. py:attribute:: LLVMConstantDataVectorValueKind
      :type:  int


   .. py:attribute:: LLVMConstantIntValueKind
      :type:  int


   .. py:attribute:: LLVMConstantByteValueKind
      :type:  int


   .. py:attribute:: LLVMConstantFPValueKind
      :type:  int


   .. py:attribute:: LLVMConstantPointerNullValueKind
      :type:  int


   .. py:attribute:: LLVMConstantTokenNoneValueKind
      :type:  int


   .. py:attribute:: LLVMMetadataAsValueValueKind
      :type:  int


   .. py:attribute:: LLVMInlineAsmValueKind
      :type:  int


   .. py:attribute:: LLVMInstructionValueKind
      :type:  int


   .. py:attribute:: LLVMPoisonValueValueKind
      :type:  int


   .. py:attribute:: LLVMConstantTargetNoneValueKind
      :type:  int


   .. py:attribute:: LLVMConstantPtrAuthValueKind
      :type:  int


.. py:class:: LLVMIntPredicate

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMIntEQ
      :type:  int


   .. py:attribute:: LLVMIntNE
      :type:  int


   .. py:attribute:: LLVMIntUGT
      :type:  int


   .. py:attribute:: LLVMIntUGE
      :type:  int


   .. py:attribute:: LLVMIntULT
      :type:  int


   .. py:attribute:: LLVMIntULE
      :type:  int


   .. py:attribute:: LLVMIntSGT
      :type:  int


   .. py:attribute:: LLVMIntSGE
      :type:  int


   .. py:attribute:: LLVMIntSLT
      :type:  int


   .. py:attribute:: LLVMIntSLE
      :type:  int


.. py:class:: LLVMRealPredicate

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMRealPredicateFalse
      :type:  int


   .. py:attribute:: LLVMRealOEQ
      :type:  int


   .. py:attribute:: LLVMRealOGT
      :type:  int


   .. py:attribute:: LLVMRealOGE
      :type:  int


   .. py:attribute:: LLVMRealOLT
      :type:  int


   .. py:attribute:: LLVMRealOLE
      :type:  int


   .. py:attribute:: LLVMRealONE
      :type:  int


   .. py:attribute:: LLVMRealORD
      :type:  int


   .. py:attribute:: LLVMRealUNO
      :type:  int


   .. py:attribute:: LLVMRealUEQ
      :type:  int


   .. py:attribute:: LLVMRealUGT
      :type:  int


   .. py:attribute:: LLVMRealUGE
      :type:  int


   .. py:attribute:: LLVMRealULT
      :type:  int


   .. py:attribute:: LLVMRealULE
      :type:  int


   .. py:attribute:: LLVMRealUNE
      :type:  int


   .. py:attribute:: LLVMRealPredicateTrue
      :type:  int


.. py:class:: LLVMThreadLocalMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMNotThreadLocal
      :type:  int


   .. py:attribute:: LLVMGeneralDynamicTLSModel
      :type:  int


   .. py:attribute:: LLVMLocalDynamicTLSModel
      :type:  int


   .. py:attribute:: LLVMInitialExecTLSModel
      :type:  int


   .. py:attribute:: LLVMLocalExecTLSModel
      :type:  int


.. py:class:: LLVMAtomicOrdering

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMAtomicOrderingNotAtomic
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingUnordered
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingMonotonic
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingAcquire
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingRelease
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingAcquireRelease
      :type:  int


   .. py:attribute:: LLVMAtomicOrderingSequentiallyConsistent
      :type:  int


.. py:class:: LLVMAtomicRMWBinOp

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMAtomicRMWBinOpXchg
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpAdd
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpSub
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpAnd
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpNand
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpOr
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpXor
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpMax
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpMin
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUMax
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUMin
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFAdd
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFSub
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMax
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMin
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUIncWrap
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUDecWrap
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUSubCond
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpUSubSat
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMaximum
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMinimum
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMaximumNum
      :type:  int


   .. py:attribute:: LLVMAtomicRMWBinOpFMinimumNum
      :type:  int


.. py:class:: LLVMDiagnosticSeverity

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMDSError
      :type:  int


   .. py:attribute:: LLVMDSWarning
      :type:  int


   .. py:attribute:: LLVMDSRemark
      :type:  int


   .. py:attribute:: LLVMDSNote
      :type:  int


.. py:class:: LLVMInlineAsmDialect

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMInlineAsmDialectATT
      :type:  int


   .. py:attribute:: LLVMInlineAsmDialectIntel
      :type:  int


.. py:class:: LLVMModuleFlagBehavior

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMModuleFlagBehaviorError
      :type:  int


   .. py:attribute:: LLVMModuleFlagBehaviorWarning
      :type:  int


   .. py:attribute:: LLVMModuleFlagBehaviorRequire
      :type:  int


   .. py:attribute:: LLVMModuleFlagBehaviorOverride
      :type:  int


   .. py:attribute:: LLVMModuleFlagBehaviorAppend
      :type:  int


   .. py:attribute:: LLVMModuleFlagBehaviorAppendUnique
      :type:  int


.. py:data:: LLVMAttributeReturnIndex
   :type:  int

.. py:data:: LLVMAttributeFunctionIndex
   :type:  int

.. py:class:: LLVMTailCallKind

   Bases: :py:obj:`enum.IntEnum`


   Tail call kind for LLVMSetTailCallKind and LLVMGetTailCallKind.

   Note that 'musttail' implies 'tail'.

   See:
       CallInst::TailCallKind


   .. py:attribute:: LLVMTailCallKindNone
      :type:  int


   .. py:attribute:: LLVMTailCallKindTail
      :type:  int


   .. py:attribute:: LLVMTailCallKindMustTail
      :type:  int


   .. py:attribute:: LLVMTailCallKindNoTail
      :type:  int


.. py:data:: LLVMFastMathAllowReassoc
   :type:  int

.. py:data:: LLVMFastMathNoNaNs
   :type:  int

.. py:data:: LLVMFastMathNoInfs
   :type:  int

.. py:data:: LLVMFastMathNoSignedZeros
   :type:  int

.. py:data:: LLVMFastMathAllowReciprocal
   :type:  int

.. py:data:: LLVMFastMathAllowContract
   :type:  int

.. py:data:: LLVMFastMathApproxFunc
   :type:  int

.. py:data:: LLVMFastMathNone
   :type:  int

.. py:data:: LLVMFastMathAll
   :type:  int

.. py:data:: LLVMGEPFlagInBounds
   :type:  int

.. py:data:: LLVMGEPFlagNUSW
   :type:  int

.. py:data:: LLVMGEPFlagNUW
   :type:  int

.. py:class:: LLVMDbgRecordKind

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMDbgRecordLabel
      :type:  int


   .. py:attribute:: LLVMDbgRecordDeclare
      :type:  int


   .. py:attribute:: LLVMDbgRecordValue
      :type:  int


   .. py:attribute:: LLVMDbgRecordAssign
      :type:  int


.. py:function:: LLVMShutdown()

   Deallocate and destroy all ManagedStatic variables.

   See:
       llvm::llvm_shutdown

   See:
       :py:obj:`~.ManagedStatic`

   .. rubric:: C signature

   .. code-block:: c

       void LLVMShutdown()


.. py:function:: LLVMGetVersion()

   Return the major, minor, and patch version of LLVM

   The version components are returned via the function's three output
   parameters or skipped if a NULL pointer was supplied.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * Major (:py:obj:`~.int`):
           (undocumented)
       * Minor (:py:obj:`~.int`):
           (undocumented)
       * Patch (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetVersion(unsigned int * Major, unsigned int * Minor, unsigned int * Patch)


.. py:function:: LLVMCreateMessage(Message)

   ===-- Error handling ----------------------------------------------------===

   Args:
       Message (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMCreateMessage(const char * Message)


.. py:function:: LLVMDisposeMessage(Message)

   ===-- Error handling ----------------------------------------------------===

   Args:
       Message (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeMessage(char * Message)


.. py:class:: LLVMDiagnosticHandler(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Contexts are execution states for the core LLVM IR system.

   Most types are tied to a context instance. Multiple contexts can
   exist simultaneously. A single context is not thread safe. However,
   different contexts can execute on different threads simultaneously.


.. py:class:: LLVMYieldCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: LLVMContextCreate()

   Create a new context.

   Every call to this function should be paired with a call to
   LLVMContextDispose() or the context will leak memory.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMContextCreate()


.. py:function:: LLVMGetGlobalContext()

   Obtain the global context instance.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMGetGlobalContext()


.. py:function:: LLVMContextSetDiagnosticHandler(C, Handler, DiagnosticContext)

   Set the diagnostic handler for this context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Handler (:py:obj:`~.LLVMDiagnosticHandler`/:py:obj:`~.object`):
           (undocumented)

       DiagnosticContext (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMContextSetDiagnosticHandler(LLVMContextRef C, LLVMDiagnosticHandler Handler, void * DiagnosticContext)


.. py:function:: LLVMContextGetDiagnosticHandler(C)

   Get the diagnostic handler of this context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDiagnosticHandler LLVMContextGetDiagnosticHandler(LLVMContextRef C)


.. py:function:: LLVMContextGetDiagnosticContext(C)

   Get the diagnostic context of this context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void * LLVMContextGetDiagnosticContext(LLVMContextRef C)


.. py:function:: LLVMContextSetYieldCallback(C, Callback, OpaqueHandle)

   Set the yield callback function for this context.

   See:
       LLVMContext::setYieldCallback()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Callback (:py:obj:`~.LLVMYieldCallback`/:py:obj:`~.object`):
           (undocumented)

       OpaqueHandle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMContextSetYieldCallback(LLVMContextRef C, LLVMYieldCallback Callback, void * OpaqueHandle)


.. py:function:: LLVMContextShouldDiscardValueNames(C)

   Retrieve whether the given context is set to discard all value names.

   See:
       LLVMContext::shouldDiscardValueNames()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMContextShouldDiscardValueNames(LLVMContextRef C)


.. py:function:: LLVMContextSetDiscardValueNames(C, Discard)

   Set whether the given context discards all value names.

   If true, only the names of GlobalValue objects will be available in the IR.
   This can be used to save memory and runtime, especially in release mode.

   See:
       LLVMContext::setDiscardValueNames()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Discard (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMContextSetDiscardValueNames(LLVMContextRef C, LLVMBool Discard)


.. py:function:: LLVMContextDispose(C)

   Destroy a context instance.

   This should be called for every call to LLVMContextCreate() or memory
   will be leaked.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMContextDispose(LLVMContextRef C)


.. py:function:: LLVMGetDiagInfoDescription(DI)

   Return a string representation of the DiagnosticInfo.

   Use
   LLVMDisposeMessage to free the string.

   See:
       DiagnosticInfo::print()

   Args:
       DI (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetDiagInfoDescription(LLVMDiagnosticInfoRef DI)


.. py:function:: LLVMGetDiagInfoSeverity(DI)

   Return an enum LLVMDiagnosticSeverity.

   See:
       DiagnosticInfo::getSeverity()

   Args:
       DI (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMDiagnosticSeverity`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDiagnosticSeverity LLVMGetDiagInfoSeverity(LLVMDiagnosticInfoRef DI)


.. py:function:: LLVMGetMDKindIDInContext(C, Name, SLen)

   Return an enum LLVMDiagnosticSeverity.

   See:
       DiagnosticInfo::getSeverity()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetMDKindIDInContext(LLVMContextRef C, const char * Name, unsigned int SLen)


.. py:function:: LLVMGetMDKindID(Name, SLen)

   (No short description, might be part of a group.)

   Args:
       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetMDKindID(const char * Name, unsigned int SLen)


.. py:function:: LLVMGetSyncScopeID(C, Name, SLen)

   Maps a synchronization scope name to a ID unique within this context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetSyncScopeID(LLVMContextRef C, const char * Name, size_t SLen)


.. py:function:: LLVMGetEnumAttributeKindForName(Name, SLen)

   Return an unique id given the name of a enum attribute,
   or 0 if no attribute by that name exists.

   See http://llvm.org/docs/LangRef.html:py:obj:`~.parameter`-attributes
   and http://llvm.org/docs/LangRef.html:py:obj:`~.function`-attributes
   for the list of available attributes.

   NB: Attribute names and/or id are subject to change without
   going through the C API deprecation cycle.

   Args:
       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetEnumAttributeKindForName(const char * Name, size_t SLen)


.. py:function:: LLVMGetLastEnumAttributeKind()

   Return an unique id given the name of a enum attribute,
   or 0 if no attribute by that name exists.

   See http://llvm.org/docs/LangRef.html:py:obj:`~.parameter`-attributes
   and http://llvm.org/docs/LangRef.html:py:obj:`~.function`-attributes
   for the list of available attributes.

   NB: Attribute names and/or id are subject to change without
   going through the C API deprecation cycle.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetLastEnumAttributeKind()


.. py:function:: LLVMCreateEnumAttribute(C, KindID, Val)

   Create an enum attribute.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

       Val (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMCreateEnumAttribute(LLVMContextRef C, unsigned int KindID, uint64_t Val)


.. py:function:: LLVMGetEnumAttributeKind(A)

   Get the unique id corresponding to the enum attribute
   passed as argument.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetEnumAttributeKind(LLVMAttributeRef A)


.. py:function:: LLVMGetEnumAttributeValue(A)

   Get the enum attribute's value. 0 is returned if none exists.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetEnumAttributeValue(LLVMAttributeRef A)


.. py:function:: LLVMCreateTypeAttribute(C, KindID, type_ref)

   Create a type attribute

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

       type_ref (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMCreateTypeAttribute(LLVMContextRef C, unsigned int KindID, LLVMTypeRef type_ref)


.. py:function:: LLVMGetTypeAttributeValue(A)

   Get the type attribute's value.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetTypeAttributeValue(LLVMAttributeRef A)


.. py:function:: LLVMCreateConstantRangeAttribute(C, KindID, NumBits, LowerWords, UpperWords)

   Create a ConstantRange attribute.

   LowerWords and UpperWords need to be NumBits divided by 64 rounded up
   elements long.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

       NumBits (:py:obj:`~.int`):
           (undocumented)

       LowerWords (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       UpperWords (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMCreateConstantRangeAttribute(LLVMContextRef C, unsigned int KindID, unsigned int NumBits, const uint64_t[] LowerWords, const uint64_t[] UpperWords)


.. py:class:: LLVMDenormalModeKind

   Bases: :py:obj:`enum.IntEnum`


   Represent different denormal handling kinds for use with
   LLVMCreateDenormalFPEnvAttribute.


   .. py:attribute:: LLVMDenormalModeKindIEEE
      :type:  int


   .. py:attribute:: LLVMDenormalModeKindPreserveSign
      :type:  int


   .. py:attribute:: LLVMDenormalModeKindPositiveZero
      :type:  int


   .. py:attribute:: LLVMDenormalModeKindDynamic
      :type:  int


.. py:function:: LLVMCreateDenormalFPEnvAttribute(C, DefaultModeOutput, DefaultModeInput, FloatModeOutput, FloatModeInput)

   Create a DenormalFPEnv attribute.

   ``DefaultModeOutput`` is the assumed denormal handling for the outputs of most
      floating-point types.

   ``DefaultModeInput`` is the assumed denormal handling for the inputs of most
      floating-point types.

   ``FloatModeOutput`` is the assumed denormal handling for the outputs of
      float. This should always be the same as as DefaultModeOutput for most
      targets.

   ``FloatModeInput`` is the assumed denormal handling for the inputs of
      float. This should always be the same as as DefaultModeInput for most
      targets.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DefaultModeOutput (:py:obj:`~.LLVMDenormalModeKind`):
           (undocumented)

       DefaultModeInput (:py:obj:`~.LLVMDenormalModeKind`):
           (undocumented)

       FloatModeOutput (:py:obj:`~.LLVMDenormalModeKind`):
           (undocumented)

       FloatModeInput (:py:obj:`~.LLVMDenormalModeKind`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMCreateDenormalFPEnvAttribute(LLVMContextRef C, LLVMDenormalModeKind DefaultModeOutput, LLVMDenormalModeKind DefaultModeInput, LLVMDenormalModeKind FloatModeOutput, LLVMDenormalModeKind FloatModeInput)


.. py:function:: LLVMCreateStringAttribute(C, K, KLength, V, VLength)

   Create a string attribute.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       K (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KLength (:py:obj:`~.int`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       VLength (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMCreateStringAttribute(LLVMContextRef C, const char * K, unsigned int KLength, const char * V, unsigned int VLength)


.. py:function:: LLVMGetStringAttributeKind(A, Length)

   Get the string attribute's kind.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetStringAttributeKind(LLVMAttributeRef A, unsigned int * Length)


.. py:function:: LLVMGetStringAttributeValue(A, Length)

   Get the string attribute's value.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetStringAttributeValue(LLVMAttributeRef A, unsigned int * Length)


.. py:function:: LLVMIsEnumAttribute(A)

   Check for the different types of attributes.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsEnumAttribute(LLVMAttributeRef A)


.. py:function:: LLVMIsStringAttribute(A)

   Check for the different types of attributes.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsStringAttribute(LLVMAttributeRef A)


.. py:function:: LLVMIsTypeAttribute(A)

   Check for the different types of attributes.

   Args:
       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsTypeAttribute(LLVMAttributeRef A)


.. py:function:: LLVMGetTypeByName2(C, Name)

   Obtain a Type from a context by its registered name.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetTypeByName2(LLVMContextRef C, const char * Name)


.. py:function:: LLVMModuleCreateWithName(ModuleID)

   Create a new, empty module in the global context.

   This is equivalent to calling LLVMModuleCreateWithNameInContext with
   LLVMGetGlobalContext() as the context parameter.

   Every invocation should be paired with LLVMDisposeModule() or memory
   will be leaked.

   Args:
       ModuleID (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleRef LLVMModuleCreateWithName(const char * ModuleID)


.. py:function:: LLVMModuleCreateWithNameInContext(ModuleID, C)

   Create a new, empty module in a specific context.

   Every invocation should be paired with LLVMDisposeModule() or memory
   will be leaked.

   Args:
       ModuleID (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleRef LLVMModuleCreateWithNameInContext(const char * ModuleID, LLVMContextRef C)


.. py:function:: LLVMCloneModule(M)

   Return an exact copy of the specified module.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleRef LLVMCloneModule(LLVMModuleRef M)


.. py:function:: LLVMDisposeModule(M)

   Destroy a module instance.

   This must be called for every created module or memory will be
   leaked.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeModule(LLVMModuleRef M)


.. py:function:: LLVMIsNewDbgInfoFormat(M)

   Soon to be deprecated.

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   Returns true if the module is in the new debug info mode which uses
   non-instruction debug records instead of debug intrinsics for variable
   location tracking.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsNewDbgInfoFormat(LLVMModuleRef M)


.. py:function:: LLVMSetIsNewDbgInfoFormat(M, UseNewFormat)

   Soon to be deprecated.

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   Convert module into desired debug info format.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       UseNewFormat (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetIsNewDbgInfoFormat(LLVMModuleRef M, LLVMBool UseNewFormat)


.. py:function:: LLVMGetModuleIdentifier(M, Len)

   Obtain the identifier of a module.

   See:
       Module::getModuleIdentifier()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Module to obtain identifier of

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           Out parameter which holds the length of the returned string.

   Returns:
       :py:obj:`~.bytes`: The identifier of M.

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetModuleIdentifier(LLVMModuleRef M, size_t * Len)


.. py:function:: LLVMSetModuleIdentifier(M, Ident, Len)

   Set the identifier of a module to a string Ident with length Len.

   See:
       Module::setModuleIdentifier()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The module to set identifier

       Ident (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The string to set M's identifier to

       Len (:py:obj:`~.int`):
           Length of Ident

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetModuleIdentifier(LLVMModuleRef M, const char * Ident, size_t Len)


.. py:function:: LLVMGetSourceFileName(M, Len)

   Obtain the module's original source file name.

   See:
       Module::getSourceFileName()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Module to obtain the name of

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           Out parameter which holds the length of the returned string

   Returns:
       :py:obj:`~.bytes`: The original source file name of M

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetSourceFileName(LLVMModuleRef M, size_t * Len)


.. py:function:: LLVMSetSourceFileName(M, Name, Len)

   Set the original source file name of a module to a string Name with length
   Len.

   See:
       Module::setSourceFileName()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The module to set the source file name of

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The string to set M's source file name to

       Len (:py:obj:`~.int`):
           Length of Name

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetSourceFileName(LLVMModuleRef M, const char * Name, size_t Len)


.. py:function:: LLVMGetDataLayoutStr(M)

   Obtain the data layout for a module.

   See:
       Module::getDataLayoutStr()

   LLVMGetDataLayout is DEPRECATED, as the name is not only incorrect,
   but match the name of another method on the module. Prefer the use
   of LLVMGetDataLayoutStr, which is not ambiguous.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetDataLayoutStr(LLVMModuleRef M)


.. py:function:: LLVMGetDataLayout(M)

   Obtain the data layout for a module.

   See:
       Module::getDataLayoutStr()

   LLVMGetDataLayout is DEPRECATED, as the name is not only incorrect,
   but match the name of another method on the module. Prefer the use
   of LLVMGetDataLayoutStr, which is not ambiguous.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetDataLayout(LLVMModuleRef M)


.. py:function:: LLVMSetDataLayout(M, DataLayoutStr)

   Set the data layout for a module.

   See:
       Module::setDataLayout()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DataLayoutStr (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetDataLayout(LLVMModuleRef M, const char * DataLayoutStr)


.. py:function:: LLVMGetTarget(M)

   Obtain the target triple for a module.

   See:
       Module::getTargetTriple()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetTarget(LLVMModuleRef M)


.. py:function:: LLVMSetTarget(M, Triple)

   Set the target triple for a module.

   See:
       Module::setTargetTriple()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTarget(LLVMModuleRef M, const char * Triple)


.. py:function:: LLVMCopyModuleFlagsMetadata(M, Len)

   Returns the module flags as an array of flag-key-value triples.

   The caller
   is responsible for freeing this array by calling
   ``LLVMDisposeModuleFlagsMetadata.``

   See:
       Module::getModuleFlagsMetadata()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleFlagEntry * LLVMCopyModuleFlagsMetadata(LLVMModuleRef M, size_t * Len)


.. py:function:: LLVMDisposeModuleFlagsMetadata(Entries)

   Destroys module flags metadata entries.

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeModuleFlagsMetadata(LLVMModuleFlagEntry * Entries)


.. py:function:: LLVMModuleFlagEntriesGetFlagBehavior(Entries, Index)

   Returns the flag behavior for a module flag entry at a specific index.

   See:
       Module::ModuleFlagEntry::Behavior

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMModuleFlagBehavior`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleFlagBehavior LLVMModuleFlagEntriesGetFlagBehavior(LLVMModuleFlagEntry * Entries, unsigned int Index)


.. py:function:: LLVMModuleFlagEntriesGetKey(Entries, Index, Len)

   Returns the key for a module flag entry at a specific index.

   See:
       Module::ModuleFlagEntry::Key

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMModuleFlagEntriesGetKey(LLVMModuleFlagEntry * Entries, unsigned int Index, size_t * Len)


.. py:function:: LLVMModuleFlagEntriesGetMetadata(Entries, Index)

   Returns the metadata for a module flag entry at a specific index.

   See:
       Module::ModuleFlagEntry::Val

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMModuleFlagEntriesGetMetadata(LLVMModuleFlagEntry * Entries, unsigned int Index)


.. py:function:: LLVMGetModuleFlag(M, Key, KeyLen)

   Add a module-level flag to the module-level flags metadata if it doesn't
   already exist.

   See:
       Module::getModuleFlag()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Key (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KeyLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMGetModuleFlag(LLVMModuleRef M, const char * Key, size_t KeyLen)


.. py:function:: LLVMAddModuleFlag(M, Behavior, Key, KeyLen, Val)

   Add a module-level flag to the module-level flags metadata if it doesn't
   already exist.

   See:
       Module::addModuleFlag()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Behavior (:py:obj:`~.LLVMModuleFlagBehavior`):
           (undocumented)

       Key (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KeyLen (:py:obj:`~.int`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddModuleFlag(LLVMModuleRef M, LLVMModuleFlagBehavior Behavior, const char * Key, size_t KeyLen, LLVMMetadataRef Val)


.. py:function:: LLVMDumpModule(M)

   Dump a representation of a module to stderr.

   See:
       Module::dump()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDumpModule(LLVMModuleRef M)


.. py:function:: LLVMPrintModuleToFile(M, Filename, ErrorMessage)

   Print a representation of a module to a file.

   The ErrorMessage needs to be
   disposed with LLVMDisposeMessage. Returns 0 on success, 1 otherwise.

   See:
       Module::print()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Filename (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       ErrorMessage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMPrintModuleToFile(LLVMModuleRef M, const char * Filename, char ** ErrorMessage)


.. py:function:: LLVMPrintModuleToString(M)

   Return a string representation of the module.

   Use
   LLVMDisposeMessage to free the string.

   See:
       Module::print()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMPrintModuleToString(LLVMModuleRef M)


.. py:function:: LLVMGetModuleInlineAsm(M, Len)

   Get inline assembly for a module.

   See:
       Module::getModuleInlineAsm()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetModuleInlineAsm(LLVMModuleRef M, size_t * Len)


.. py:function:: LLVMSetModuleInlineAsm2(M, Asm, Len)

   Set inline assembly for a module.

   See:
       Module::setModuleInlineAsm()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Asm (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetModuleInlineAsm2(LLVMModuleRef M, const char * Asm, size_t Len)


.. py:function:: LLVMAppendModuleInlineAsm(M, Asm, Len)

   Append inline assembly to a module.

   See:
       Module::appendModuleInlineAsm()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Asm (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAppendModuleInlineAsm(LLVMModuleRef M, const char * Asm, size_t Len)


.. py:function:: LLVMGetInlineAsm(Ty, AsmString, AsmStringSize, Constraints, ConstraintsSize, HasSideEffects, IsAlignStack, Dialect, CanThrow)

   Create the specified uniqued inline asm string.

   See:
       InlineAsm::get()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AsmString (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       AsmStringSize (:py:obj:`~.int`):
           (undocumented)

       Constraints (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       ConstraintsSize (:py:obj:`~.int`):
           (undocumented)

       HasSideEffects (:py:obj:`~.int`):
           (undocumented)

       IsAlignStack (:py:obj:`~.int`):
           (undocumented)

       Dialect (:py:obj:`~.LLVMInlineAsmDialect`):
           (undocumented)

       CanThrow (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetInlineAsm(LLVMTypeRef Ty, const char * AsmString, size_t AsmStringSize, const char * Constraints, size_t ConstraintsSize, LLVMBool HasSideEffects, LLVMBool IsAlignStack, LLVMInlineAsmDialect Dialect, LLVMBool CanThrow)


.. py:function:: LLVMGetInlineAsmAsmString(InlineAsmVal, Len)

   Get the template string used for an inline assembly snippet

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetInlineAsmAsmString(LLVMValueRef InlineAsmVal, size_t * Len)


.. py:function:: LLVMGetInlineAsmConstraintString(InlineAsmVal, Len)

   Get the raw constraint string for an inline assembly snippet

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetInlineAsmConstraintString(LLVMValueRef InlineAsmVal, size_t * Len)


.. py:function:: LLVMGetInlineAsmDialect(InlineAsmVal)

   Get the dialect used by the inline asm snippet

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMInlineAsmDialect`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMInlineAsmDialect LLVMGetInlineAsmDialect(LLVMValueRef InlineAsmVal)


.. py:function:: LLVMGetInlineAsmFunctionType(InlineAsmVal)

   Get the function type of the inline assembly snippet.

   The same type that
   was passed into LLVMGetInlineAsm originally

   See:
       :py:obj:`~.LLVMGetInlineAsm`

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetInlineAsmFunctionType(LLVMValueRef InlineAsmVal)


.. py:function:: LLVMGetInlineAsmHasSideEffects(InlineAsmVal)

   Get if the inline asm snippet has side effects

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetInlineAsmHasSideEffects(LLVMValueRef InlineAsmVal)


.. py:function:: LLVMGetInlineAsmNeedsAlignedStack(InlineAsmVal)

   Get if the inline asm snippet needs an aligned stack

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetInlineAsmNeedsAlignedStack(LLVMValueRef InlineAsmVal)


.. py:function:: LLVMGetInlineAsmCanUnwind(InlineAsmVal)

   Get if the inline asm snippet may unwind the stack

   Args:
       InlineAsmVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetInlineAsmCanUnwind(LLVMValueRef InlineAsmVal)


.. py:function:: LLVMGetModuleContext(M)

   Obtain the context to which this module is associated.

   See:
       Module::getContext()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMGetModuleContext(LLVMModuleRef M)


.. py:function:: LLVMGetTypeByName(M, Name)

   Deprecated: Use LLVMGetTypeByName2 instead.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, const char * Name)


.. py:function:: LLVMGetFirstNamedMetadata(M)

   Obtain an iterator to the first NamedMDNode in a Module.

   See:
       llvm::Module::named_metadata_begin()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetFirstNamedMetadata(LLVMModuleRef M)


.. py:function:: LLVMGetLastNamedMetadata(M)

   Obtain an iterator to the last NamedMDNode in a Module.

   See:
       llvm::Module::named_metadata_end()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetLastNamedMetadata(LLVMModuleRef M)


.. py:function:: LLVMGetNextNamedMetadata(NamedMDNode)

   Advance a NamedMDNode iterator to the next NamedMDNode.

   Returns NULL if the iterator was already at the end and there are no more
   named metadata nodes.

   Args:
       NamedMDNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetNextNamedMetadata(LLVMNamedMDNodeRef NamedMDNode)


.. py:function:: LLVMGetPreviousNamedMetadata(NamedMDNode)

   Decrement a NamedMDNode iterator to the previous NamedMDNode.

   Returns NULL if the iterator was already at the beginning and there are
   no previous named metadata nodes.

   Args:
       NamedMDNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetPreviousNamedMetadata(LLVMNamedMDNodeRef NamedMDNode)


.. py:function:: LLVMGetNamedMetadata(M, Name, NameLen)

   Retrieve a NamedMDNode with the given name, returning NULL if no such
   node exists.

   See:
       llvm::Module::getNamedMetadata()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetNamedMetadata(LLVMModuleRef M, const char * Name, size_t NameLen)


.. py:function:: LLVMGetOrInsertNamedMetadata(M, Name, NameLen)

   Retrieve a NamedMDNode with the given name, creating a new node if no such
   node exists.

   See:
       llvm::Module::getOrInsertNamedMetadata()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMNamedMDNodeRef LLVMGetOrInsertNamedMetadata(LLVMModuleRef M, const char * Name, size_t NameLen)


.. py:function:: LLVMGetNamedMetadataName(NamedMD, NameLen)

   Retrieve the name of a NamedMDNode.

   See:
       llvm::NamedMDNode::getName()

   Args:
       NamedMD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetNamedMetadataName(LLVMNamedMDNodeRef NamedMD, size_t * NameLen)


.. py:function:: LLVMGetNamedMetadataNumOperands(M, Name)

   Obtain the number of operands for named metadata in a module.

   See:
       llvm::Module::getNamedMetadata()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNamedMetadataNumOperands(LLVMModuleRef M, const char * Name)


.. py:function:: LLVMGetNamedMetadataOperands(M, Name, Dest)

   Obtain the named metadata operands for a module.

   The passed LLVMValueRef pointer should refer to an array of
   LLVMValueRef at least LLVMGetNamedMetadataNumOperands long. This
   array will be populated with the LLVMValueRef instances. Each
   instance corresponds to a llvm::MDNode.

   See:
       llvm::Module::getNamedMetadata()

   See:
       llvm::MDNode::getOperand()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetNamedMetadataOperands(LLVMModuleRef M, const char * Name, LLVMValueRef * Dest)


.. py:function:: LLVMAddNamedMetadataOperand(M, Name, Val)

   Add an operand to named metadata.

   See:
       llvm::Module::getNamedMetadata()

   See:
       llvm::MDNode::addOperand()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddNamedMetadataOperand(LLVMModuleRef M, const char * Name, LLVMValueRef Val)


.. py:function:: LLVMGetDebugLocDirectory(Val, Length)

   Return the directory of the debug location for this value, which must be
   an llvm::Instruction, llvm::GlobalVariable, or llvm::Function.

   See:
       llvm::Instruction::getDebugLoc()

   See:
       llvm::GlobalVariable::getDebugInfo()

   See:
       llvm::Function::getSubprogram()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetDebugLocDirectory(LLVMValueRef Val, unsigned int * Length)


.. py:function:: LLVMGetDebugLocFilename(Val, Length)

   Return the filename of the debug location for this value, which must be
   an llvm::Instruction, llvm::GlobalVariable, or llvm::Function.

   See:
       llvm::Instruction::getDebugLoc()

   See:
       llvm::GlobalVariable::getDebugInfo()

   See:
       llvm::Function::getSubprogram()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetDebugLocFilename(LLVMValueRef Val, unsigned int * Length)


.. py:function:: LLVMGetDebugLocLine(Val)

   Return the line number of the debug location for this value, which must be
   an llvm::Instruction, llvm::GlobalVariable, or llvm::Function.

   See:
       llvm::Instruction::getDebugLoc()

   See:
       llvm::GlobalVariable::getDebugInfo()

   See:
       llvm::Function::getSubprogram()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetDebugLocLine(LLVMValueRef Val)


.. py:function:: LLVMGetDebugLocColumn(Val)

   Return the column number of the debug location for this value, which must be
   an llvm::Instruction.

   See:
       llvm::Instruction::getDebugLoc()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetDebugLocColumn(LLVMValueRef Val)


.. py:function:: LLVMAddFunction(M, Name, FunctionTy)

   Add a function to a module under a specified name.

   See:
       llvm::Function::Create()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char * Name, LLVMTypeRef FunctionTy)


.. py:function:: LLVMGetOrInsertFunction(M, Name, NameLen, FunctionTy)

   Obtain or insert a function into a module.

   If a function with the specified name already exists in the module, it
   is returned. Otherwise, a new function is created in the module with the
   specified name and type and is returned.

   The returned value corresponds to a llvm::Function instance.

   See:
       llvm::Module::getOrInsertFunction()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetOrInsertFunction(LLVMModuleRef M, const char * Name, size_t NameLen, LLVMTypeRef FunctionTy)


.. py:function:: LLVMGetNamedFunction(M, Name)

   Obtain a Function value from a Module by its name.

   The returned value corresponds to a llvm::Function value.

   See:
       llvm::Module::getFunction()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, const char * Name)


.. py:function:: LLVMGetNamedFunctionWithLength(M, Name, Length)

   Obtain a Function value from a Module by its name.

   The returned value corresponds to a llvm::Function value.

   See:
       llvm::Module::getFunction()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedFunctionWithLength(LLVMModuleRef M, const char * Name, size_t Length)


.. py:function:: LLVMGetFirstFunction(M)

   Obtain an iterator to the first Function in a Module.

   See:
       llvm::Module::begin()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M)


.. py:function:: LLVMGetLastFunction(M)

   Obtain an iterator to the last Function in a Module.

   See:
       llvm::Module::end()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M)


.. py:function:: LLVMGetNextFunction(Fn)

   Advance a Function iterator to the next Function.

   Returns NULL if the iterator was already at the end and there are no more
   functions.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn)


.. py:function:: LLVMGetPreviousFunction(Fn)

   Decrement a Function iterator to the previous Function.

   Returns NULL if the iterator was already at the beginning and there are
   no previous functions.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn)


.. py:function:: LLVMSetModuleInlineAsm(M, Asm)

   Deprecated: Use LLVMSetModuleInlineAsm2 instead.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Asm (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetModuleInlineAsm(LLVMModuleRef M, const char * Asm)


.. py:function:: LLVMGetTypeKind(Ty)

   Obtain the enumerated type of a Type instance.

   See:
       llvm::Type::py:obj:`~.getTypeID`

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMTypeKind`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty)


.. py:function:: LLVMTypeIsSized(Ty)

   Whether the type has a known size.

   Things that don't have a size are abstract types, labels, and void.a

   See:
       llvm::Type::isSized()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTypeIsSized(LLVMTypeRef Ty)


.. py:function:: LLVMGetTypeContext(Ty)

   Obtain the context to which this type instance is associated.

   See:
       llvm::Type::getContext()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMGetTypeContext(LLVMTypeRef Ty)


.. py:function:: LLVMDumpType(Val)

   Dump a representation of a type to stderr.

   See:
       llvm::Type::dump()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDumpType(LLVMTypeRef Val)


.. py:function:: LLVMPrintTypeToString(Val)

   Return a string representation of the type.

   Use
   LLVMDisposeMessage to free the string.

   See:
       llvm::Type::print()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMPrintTypeToString(LLVMTypeRef Val)


.. py:function:: LLVMByteTypeInContext(C, NumBits)

   Obtain a byte type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumBits (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMByteTypeInContext(LLVMContextRef C, unsigned int NumBits)


.. py:function:: LLVMGetByteTypeWidth(ByteTy)

   Obtain a byte type from a context with specified bit width.

   Args:
       ByteTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetByteTypeWidth(LLVMTypeRef ByteTy)


.. py:function:: LLVMInt1TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt1TypeInContext(LLVMContextRef C)


.. py:function:: LLVMInt8TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt8TypeInContext(LLVMContextRef C)


.. py:function:: LLVMInt16TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt16TypeInContext(LLVMContextRef C)


.. py:function:: LLVMInt32TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt32TypeInContext(LLVMContextRef C)


.. py:function:: LLVMInt64TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt64TypeInContext(LLVMContextRef C)


.. py:function:: LLVMInt128TypeInContext(C)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt128TypeInContext(LLVMContextRef C)


.. py:function:: LLVMIntTypeInContext(C, NumBits)

   Obtain an integer type from a context with specified bit width.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumBits (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntTypeInContext(LLVMContextRef C, unsigned int NumBits)


.. py:function:: LLVMInt1Type()

   Obtain an integer type from the global context with a specified bit
   width.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt1Type()


.. py:function:: LLVMInt8Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt8Type()


.. py:function:: LLVMInt16Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt16Type()


.. py:function:: LLVMInt32Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt32Type()


.. py:function:: LLVMInt64Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt64Type()


.. py:function:: LLVMInt128Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMInt128Type()


.. py:function:: LLVMIntType(NumBits)

   (No short description, might be part of a group.)

   Args:
       NumBits (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntType(unsigned int NumBits)


.. py:function:: LLVMGetIntTypeWidth(IntegerTy)

   Obtain an integer type from the global context with a specified bit
   width.

   Args:
       IntegerTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy)


.. py:function:: LLVMHalfTypeInContext(C)

   Obtain a 16-bit floating point type from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMHalfTypeInContext(LLVMContextRef C)


.. py:function:: LLVMBFloatTypeInContext(C)

   Obtain a 16-bit brain floating point type from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMBFloatTypeInContext(LLVMContextRef C)


.. py:function:: LLVMFloatTypeInContext(C)

   Obtain a 32-bit floating point type from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMFloatTypeInContext(LLVMContextRef C)


.. py:function:: LLVMDoubleTypeInContext(C)

   Obtain a 64-bit floating point type from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMDoubleTypeInContext(LLVMContextRef C)


.. py:function:: LLVMX86FP80TypeInContext(C)

   Obtain a 80-bit floating point type (X87) from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMX86FP80TypeInContext(LLVMContextRef C)


.. py:function:: LLVMFP128TypeInContext(C)

   Obtain a 128-bit floating point type (112-bit mantissa) from a
   context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMFP128TypeInContext(LLVMContextRef C)


.. py:function:: LLVMPPCFP128TypeInContext(C)

   Obtain a 128-bit floating point type (two 64-bits) from a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMPPCFP128TypeInContext(LLVMContextRef C)


.. py:function:: LLVMHalfType()

   Obtain a floating point type from the global context.

   These map to the functions in this group of the same name.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMHalfType()


.. py:function:: LLVMBFloatType()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMBFloatType()


.. py:function:: LLVMFloatType()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMFloatType()


.. py:function:: LLVMDoubleType()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMDoubleType()


.. py:function:: LLVMX86FP80Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMX86FP80Type()


.. py:function:: LLVMFP128Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMFP128Type()


.. py:function:: LLVMPPCFP128Type()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMPPCFP128Type()


.. py:function:: LLVMFunctionType(ReturnType, ParamTypes, ParamCount, IsVarArg)

   Obtain a function type consisting of a specified signature.

   The function is defined as a tuple of a return Type, a list of
   parameter types, and whether the function is variadic.

   Args:
       ReturnType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ParamTypes (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

       ParamCount (:py:obj:`~.int`):
           (undocumented)

       IsVarArg (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType, LLVMTypeRef * ParamTypes, unsigned int ParamCount, LLVMBool IsVarArg)


.. py:function:: LLVMIsFunctionVarArg(FunctionTy)

   Returns whether a function type is variadic.

   Args:
       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy)


.. py:function:: LLVMGetReturnType(FunctionTy)

   Obtain the Type this function Type returns.

   Args:
       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy)


.. py:function:: LLVMCountParamTypes(FunctionTy)

   Obtain the number of parameters this function accepts.

   Args:
       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCountParamTypes(LLVMTypeRef FunctionTy)


.. py:function:: LLVMGetParamTypes(FunctionTy, Dest)

   Obtain the types of a function's parameters.

   The Dest parameter should point to a pre-allocated array of
   LLVMTypeRef at least LLVMCountParamTypes() large. On return, the
   first LLVMCountParamTypes() entries in the array will be populated
   with LLVMTypeRef instances.

   Args:
       FunctionTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The function type to operate on.

       Dest (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           Memory address of an array to be filled with result.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef * Dest)


.. py:function:: LLVMStructTypeInContext(C, ElementTypes, ElementCount, Packed)

   Create a new structure type in a context.

   A structure is specified by a list of inner elements/types and
   whether these can be packed together.

   See:
       llvm::StructType::create()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

       Packed (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMStructTypeInContext(LLVMContextRef C, LLVMTypeRef * ElementTypes, unsigned int ElementCount, LLVMBool Packed)


.. py:function:: LLVMStructType(ElementTypes, ElementCount, Packed)

   Create a new structure type in the global context.

   See:
       llvm::StructType::create()

   Args:
       ElementTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

       Packed (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMStructType(LLVMTypeRef * ElementTypes, unsigned int ElementCount, LLVMBool Packed)


.. py:function:: LLVMStructCreateNamed(C, Name)

   Create an empty structure in a context having a specified name.

   See:
       llvm::StructType::create()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMStructCreateNamed(LLVMContextRef C, const char * Name)


.. py:function:: LLVMGetStructName(Ty)

   Obtain the name of a structure.

   See:
       llvm::StructType::getName()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetStructName(LLVMTypeRef Ty)


.. py:function:: LLVMStructSetBody(StructTy, ElementTypes, ElementCount, Packed)

   Set the contents of a structure type.

   See:
       llvm::StructType::setBody()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

       Packed (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMStructSetBody(LLVMTypeRef StructTy, LLVMTypeRef * ElementTypes, unsigned int ElementCount, LLVMBool Packed)


.. py:function:: LLVMCountStructElementTypes(StructTy)

   Get the number of elements defined inside the structure.

   See:
       llvm::StructType::getNumElements()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCountStructElementTypes(LLVMTypeRef StructTy)


.. py:function:: LLVMGetStructElementTypes(StructTy, Dest)

   Get the elements within a structure.

   The function is passed the address of a pre-allocated array of
   LLVMTypeRef at least LLVMCountStructElementTypes() long. After
   invocation, this array will be populated with the structure's
   elements. The objects in the destination array will have a lifetime
   of the structure type itself, which is the lifetime of the context it
   is contained in.

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef * Dest)


.. py:function:: LLVMStructGetTypeAtIndex(StructTy, i)

   Get the type of the element at a given index in the structure.

   See:
       llvm::StructType::getTypeAtIndex()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef StructTy, unsigned int i)


.. py:function:: LLVMIsPackedStruct(StructTy)

   Determine whether a structure is packed.

   See:
       llvm::StructType::isPacked()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsPackedStruct(LLVMTypeRef StructTy)


.. py:function:: LLVMIsOpaqueStruct(StructTy)

   Determine whether a structure is opaque.

   See:
       llvm::StructType::isOpaque()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsOpaqueStruct(LLVMTypeRef StructTy)


.. py:function:: LLVMIsLiteralStruct(StructTy)

   Determine whether a structure is literal.

   See:
       llvm::StructType::isLiteral()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsLiteralStruct(LLVMTypeRef StructTy)


.. py:function:: LLVMGetElementType(Ty)

   Obtain the element type of an array or vector type.

   See:
       llvm::SequentialType::getElementType()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty)


.. py:function:: LLVMGetSubtypes(Tp, Arr)

   Returns type's subtypes

   See:
       llvm::Type::subtypes()

   Args:
       Tp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Arr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetSubtypes(LLVMTypeRef Tp, LLVMTypeRef * Arr)


.. py:function:: LLVMGetNumContainedTypes(Tp)

   Return the number of types in the derived type.

   See:
       llvm::Type::getNumContainedTypes()

   Args:
       Tp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumContainedTypes(LLVMTypeRef Tp)


.. py:function:: LLVMArrayType(ElementType, ElementCount)

   Create a fixed size array type that refers to a specific type.

   The created type will exist in the context that its element type
   exists in.

   Deprecated:
       LLVMArrayType is deprecated in favor of the API accurate
       LLVMArrayType2

   See:
       llvm::ArrayType::get()

   Args:
       ElementType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, unsigned int ElementCount)


.. py:function:: LLVMArrayType2(ElementType, ElementCount)

   Create a fixed size array type that refers to a specific type.

   The created type will exist in the context that its element type
   exists in.

   See:
       llvm::ArrayType::get()

   Args:
       ElementType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMArrayType2(LLVMTypeRef ElementType, uint64_t ElementCount)


.. py:function:: LLVMGetArrayLength(ArrayTy)

   Obtain the length of an array type.

   This only works on types that represent arrays.

   Deprecated:
       LLVMGetArrayLength is deprecated in favor of the API accurate
       LLVMGetArrayLength2

   See:
       llvm::ArrayType::getNumElements()

   Args:
       ArrayTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetArrayLength(LLVMTypeRef ArrayTy)


.. py:function:: LLVMGetArrayLength2(ArrayTy)

   Obtain the length of an array type.

   This only works on types that represent arrays.

   See:
       llvm::ArrayType::getNumElements()

   Args:
       ArrayTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetArrayLength2(LLVMTypeRef ArrayTy)


.. py:function:: LLVMPointerType(ElementType, AddressSpace)

   Create a pointer type that points to a defined type.

   The created type will exist in the context that its pointee type
   exists in.

   See:
       llvm::PointerType::get()

   Args:
       ElementType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AddressSpace (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, unsigned int AddressSpace)


.. py:function:: LLVMPointerTypeIsOpaque(Ty)

   Determine whether a pointer is opaque.

   True if this is an instance of an opaque PointerType.

   See:
       llvm::Type::isOpaquePointerTy()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMPointerTypeIsOpaque(LLVMTypeRef Ty)


.. py:function:: LLVMPointerTypeInContext(C, AddressSpace)

   Create an opaque pointer type in a context.

   See:
       llvm::PointerType::get()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AddressSpace (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMPointerTypeInContext(LLVMContextRef C, unsigned int AddressSpace)


.. py:function:: LLVMGetPointerAddressSpace(PointerTy)

   Obtain the address space of a pointer type.

   This only works on types that represent pointers.

   See:
       llvm::PointerType::getAddressSpace()

   Args:
       PointerTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy)


.. py:function:: LLVMVectorType(ElementType, ElementCount)

   Create a vector type that contains a defined type and has a specific
   number of elements.

   The created type will exist in the context thats its element type
   exists in.

   See:
       llvm::VectorType::get()

   Args:
       ElementType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, unsigned int ElementCount)


.. py:function:: LLVMScalableVectorType(ElementType, ElementCount)

   Create a vector type that contains a defined type and has a scalable
   number of elements.

   The created type will exist in the context thats its element type
   exists in.

   See:
       llvm::ScalableVectorType::get()

   Args:
       ElementType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMScalableVectorType(LLVMTypeRef ElementType, unsigned int ElementCount)


.. py:function:: LLVMGetVectorSize(VectorTy)

   Obtain the (possibly scalable) number of elements in a vector type.

   This only works on types that represent vectors (fixed or scalable).

   See:
       llvm::VectorType::getNumElements()

   Args:
       VectorTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetVectorSize(LLVMTypeRef VectorTy)


.. py:function:: LLVMGetConstantPtrAuthPointer(PtrAuth)

   Get the pointer value for the associated ConstantPtrAuth constant.

   See:
       llvm::ConstantPtrAuth::getPointer

   Args:
       PtrAuth (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetConstantPtrAuthPointer(LLVMValueRef PtrAuth)


.. py:function:: LLVMGetConstantPtrAuthKey(PtrAuth)

   Get the key value for the associated ConstantPtrAuth constant.

   See:
       llvm::ConstantPtrAuth::getKey

   Args:
       PtrAuth (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetConstantPtrAuthKey(LLVMValueRef PtrAuth)


.. py:function:: LLVMGetConstantPtrAuthDiscriminator(PtrAuth)

   Get the discriminator value for the associated ConstantPtrAuth constant.

   See:
       llvm::ConstantPtrAuth::getDiscriminator

   Args:
       PtrAuth (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetConstantPtrAuthDiscriminator(LLVMValueRef PtrAuth)


.. py:function:: LLVMGetConstantPtrAuthAddrDiscriminator(PtrAuth)

   Get the address discriminator value for the associated ConstantPtrAuth
   constant.

   See:
       llvm::ConstantPtrAuth::getAddrDiscriminator

   Args:
       PtrAuth (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetConstantPtrAuthAddrDiscriminator(LLVMValueRef PtrAuth)


.. py:function:: LLVMVoidTypeInContext(C)

   Create a void type in a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMVoidTypeInContext(LLVMContextRef C)


.. py:function:: LLVMLabelTypeInContext(C)

   Create a label type in a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMLabelTypeInContext(LLVMContextRef C)


.. py:function:: LLVMX86AMXTypeInContext(C)

   Create a X86 AMX type in a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMX86AMXTypeInContext(LLVMContextRef C)


.. py:function:: LLVMTokenTypeInContext(C)

   Create a token type in a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMTokenTypeInContext(LLVMContextRef C)


.. py:function:: LLVMMetadataTypeInContext(C)

   Create a metadata type in a context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMMetadataTypeInContext(LLVMContextRef C)


.. py:function:: LLVMVoidType()

   These are similar to the above functions except they operate on the
   global context.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMVoidType()


.. py:function:: LLVMLabelType()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMLabelType()


.. py:function:: LLVMX86AMXType()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMX86AMXType()


.. py:function:: LLVMTargetExtTypeInContext(C, Name, TypeParams, TypeParamCount, IntParams, IntParamCount)

   Create a target extension type in LLVM context.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       TypeParams (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TypeParamCount (:py:obj:`~.int`):
           (undocumented)

       IntParams (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

       IntParamCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMTargetExtTypeInContext(LLVMContextRef C, const char * Name, LLVMTypeRef * TypeParams, unsigned int TypeParamCount, unsigned int * IntParams, unsigned int IntParamCount)


.. py:function:: LLVMGetTargetExtTypeName(TargetExtTy)

   Obtain the name for this target extension type.

   See:
       llvm::TargetExtType::getName()

   Args:
       TargetExtTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetTargetExtTypeName(LLVMTypeRef TargetExtTy)


.. py:function:: LLVMGetTargetExtTypeNumTypeParams(TargetExtTy)

   Obtain the number of type parameters for this target extension type.

   See:
       llvm::TargetExtType::getNumTypeParameters()

   Args:
       TargetExtTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetTargetExtTypeNumTypeParams(LLVMTypeRef TargetExtTy)


.. py:function:: LLVMGetTargetExtTypeTypeParam(TargetExtTy, Idx)

   Get the type parameter at the given index for the target extension type.

   See:
       llvm::TargetExtType::getTypeParameter()

   Args:
       TargetExtTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetTargetExtTypeTypeParam(LLVMTypeRef TargetExtTy, unsigned int Idx)


.. py:function:: LLVMGetTargetExtTypeNumIntParams(TargetExtTy)

   Obtain the number of int parameters for this target extension type.

   See:
       llvm::TargetExtType::getNumIntParameters()

   Args:
       TargetExtTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetTargetExtTypeNumIntParams(LLVMTypeRef TargetExtTy)


.. py:function:: LLVMGetTargetExtTypeIntParam(TargetExtTy, Idx)

   Get the int parameter at the given index for the target extension type.

   See:
       llvm::TargetExtType::getIntParameter()

   Args:
       TargetExtTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetTargetExtTypeIntParam(LLVMTypeRef TargetExtTy, unsigned int Idx)


.. py:function:: LLVMTypeOf(Val)

   Obtain the type of a value.

   See:
       llvm::Value::getType()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMTypeOf(LLVMValueRef Val)


.. py:function:: LLVMGetValueKind(Val)

   Obtain the enumerated type of a Value instance.

   See:
       llvm::Value::getValueID()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMValueKind`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueKind LLVMGetValueKind(LLVMValueRef Val)


.. py:function:: LLVMGetValueName2(Val)

   Obtain the string name of a value.

   See:
       llvm::Value::getName()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.bytes`: (undocumented)
       * Length (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetValueName2(LLVMValueRef Val, size_t * Length)


.. py:function:: LLVMSetValueName2(Val, Name, NameLen)

   Set the string name of a value.

   See:
       llvm::Value::setName()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetValueName2(LLVMValueRef Val, const char * Name, size_t NameLen)


.. py:function:: LLVMDumpValue(Val)

   Dump a representation of a value to stderr.

   See:
       llvm::Value::dump()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDumpValue(LLVMValueRef Val)


.. py:function:: LLVMPrintValueToString(Val)

   Return a string representation of the value.

   Use
   LLVMDisposeMessage to free the string.

   See:
       llvm::Value::print()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMPrintValueToString(LLVMValueRef Val)


.. py:function:: LLVMGetValueContext(Val)

   Obtain the context to which this value is associated.

   See:
       llvm::Value::getContext()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMGetValueContext(LLVMValueRef Val)


.. py:function:: LLVMPrintDbgRecordToString(Record)

   Return a string representation of the DbgRecord.

   Use
   LLVMDisposeMessage to free the string.

   See:
       llvm::DbgRecord::print()

   Args:
       Record (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMPrintDbgRecordToString(LLVMDbgRecordRef Record)


.. py:function:: LLVMReplaceAllUsesWith(OldVal, NewVal)

   Replace all uses of a value with another one.

   See:
       llvm::Value::replaceAllUsesWith()

   Args:
       OldVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NewVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMReplaceAllUsesWith(LLVMValueRef OldVal, LLVMValueRef NewVal)


.. py:function:: LLVMIsConstant(Val)

   Determine whether the specified value instance is constant.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsConstant(LLVMValueRef Val)


.. py:function:: LLVMIsUndef(Val)

   Determine whether a value instance is undefined.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsUndef(LLVMValueRef Val)


.. py:function:: LLVMIsPoison(Val)

   Determine whether a value instance is poisonous.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsPoison(LLVMValueRef Val)


.. py:function:: LLVMIsAArgument(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAArgument(LLVMValueRef Val)


.. py:function:: LLVMIsABasicBlock(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsABasicBlock(LLVMValueRef Val)


.. py:function:: LLVMIsAInlineAsm(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAInlineAsm(LLVMValueRef Val)


.. py:function:: LLVMIsAUser(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUser(LLVMValueRef Val)


.. py:function:: LLVMIsAConstant(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstant(LLVMValueRef Val)


.. py:function:: LLVMIsABlockAddress(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsABlockAddress(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantAggregateZero(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantAggregateZero(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantArray(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantArray(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantDataSequential(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantDataSequential(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantDataArray(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantDataArray(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantDataVector(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantDataVector(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantExpr(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantExpr(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantFP(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantFP(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantInt(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantInt(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantByte(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantByte(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantPointerNull(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantPointerNull(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantStruct(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantStruct(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantTokenNone(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantTokenNone(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantVector(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantVector(LLVMValueRef Val)


.. py:function:: LLVMIsAConstantPtrAuth(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAConstantPtrAuth(LLVMValueRef Val)


.. py:function:: LLVMIsAGlobalValue(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGlobalValue(LLVMValueRef Val)


.. py:function:: LLVMIsAGlobalAlias(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGlobalAlias(LLVMValueRef Val)


.. py:function:: LLVMIsAGlobalObject(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGlobalObject(LLVMValueRef Val)


.. py:function:: LLVMIsAFunction(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFunction(LLVMValueRef Val)


.. py:function:: LLVMIsAGlobalVariable(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGlobalVariable(LLVMValueRef Val)


.. py:function:: LLVMIsAGlobalIFunc(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGlobalIFunc(LLVMValueRef Val)


.. py:function:: LLVMIsAUndefValue(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUndefValue(LLVMValueRef Val)


.. py:function:: LLVMIsAPoisonValue(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAPoisonValue(LLVMValueRef Val)


.. py:function:: LLVMIsAInstruction(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAInstruction(LLVMValueRef Val)


.. py:function:: LLVMIsAUnaryOperator(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUnaryOperator(LLVMValueRef Val)


.. py:function:: LLVMIsABinaryOperator(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsABinaryOperator(LLVMValueRef Val)


.. py:function:: LLVMIsACallInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACallInst(LLVMValueRef Val)


.. py:function:: LLVMIsAIntrinsicInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAIntrinsicInst(LLVMValueRef Val)


.. py:function:: LLVMIsADbgInfoIntrinsic(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsADbgInfoIntrinsic(LLVMValueRef Val)


.. py:function:: LLVMIsADbgVariableIntrinsic(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsADbgVariableIntrinsic(LLVMValueRef Val)


.. py:function:: LLVMIsADbgDeclareInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsADbgDeclareInst(LLVMValueRef Val)


.. py:function:: LLVMIsADbgLabelInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsADbgLabelInst(LLVMValueRef Val)


.. py:function:: LLVMIsAMemIntrinsic(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMemIntrinsic(LLVMValueRef Val)


.. py:function:: LLVMIsAMemCpyInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMemCpyInst(LLVMValueRef Val)


.. py:function:: LLVMIsAMemMoveInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMemMoveInst(LLVMValueRef Val)


.. py:function:: LLVMIsAMemSetInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMemSetInst(LLVMValueRef Val)


.. py:function:: LLVMIsACmpInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACmpInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFCmpInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFCmpInst(LLVMValueRef Val)


.. py:function:: LLVMIsAICmpInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAICmpInst(LLVMValueRef Val)


.. py:function:: LLVMIsAExtractElementInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAExtractElementInst(LLVMValueRef Val)


.. py:function:: LLVMIsAGetElementPtrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAGetElementPtrInst(LLVMValueRef Val)


.. py:function:: LLVMIsAInsertElementInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAInsertElementInst(LLVMValueRef Val)


.. py:function:: LLVMIsAInsertValueInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAInsertValueInst(LLVMValueRef Val)


.. py:function:: LLVMIsALandingPadInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsALandingPadInst(LLVMValueRef Val)


.. py:function:: LLVMIsAPHINode(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAPHINode(LLVMValueRef Val)


.. py:function:: LLVMIsASelectInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsASelectInst(LLVMValueRef Val)


.. py:function:: LLVMIsAShuffleVectorInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAShuffleVectorInst(LLVMValueRef Val)


.. py:function:: LLVMIsAStoreInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAStoreInst(LLVMValueRef Val)


.. py:function:: LLVMIsAUncondBrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUncondBrInst(LLVMValueRef Val)


.. py:function:: LLVMIsACondBrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACondBrInst(LLVMValueRef Val)


.. py:function:: LLVMIsAIndirectBrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAIndirectBrInst(LLVMValueRef Val)


.. py:function:: LLVMIsAInvokeInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAInvokeInst(LLVMValueRef Val)


.. py:function:: LLVMIsAReturnInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAReturnInst(LLVMValueRef Val)


.. py:function:: LLVMIsASwitchInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsASwitchInst(LLVMValueRef Val)


.. py:function:: LLVMIsAUnreachableInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUnreachableInst(LLVMValueRef Val)


.. py:function:: LLVMIsAResumeInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAResumeInst(LLVMValueRef Val)


.. py:function:: LLVMIsACleanupReturnInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACleanupReturnInst(LLVMValueRef Val)


.. py:function:: LLVMIsACatchReturnInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACatchReturnInst(LLVMValueRef Val)


.. py:function:: LLVMIsACatchSwitchInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACatchSwitchInst(LLVMValueRef Val)


.. py:function:: LLVMIsACallBrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACallBrInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFuncletPadInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFuncletPadInst(LLVMValueRef Val)


.. py:function:: LLVMIsACatchPadInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACatchPadInst(LLVMValueRef Val)


.. py:function:: LLVMIsACleanupPadInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACleanupPadInst(LLVMValueRef Val)


.. py:function:: LLVMIsAUnaryInstruction(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUnaryInstruction(LLVMValueRef Val)


.. py:function:: LLVMIsAAllocaInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAAllocaInst(LLVMValueRef Val)


.. py:function:: LLVMIsACastInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsACastInst(LLVMValueRef Val)


.. py:function:: LLVMIsAAddrSpaceCastInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAAddrSpaceCastInst(LLVMValueRef Val)


.. py:function:: LLVMIsABitCastInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsABitCastInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFPExtInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFPExtInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFPToSIInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFPToSIInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFPToUIInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFPToUIInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFPTruncInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFPTruncInst(LLVMValueRef Val)


.. py:function:: LLVMIsAIntToPtrInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAIntToPtrInst(LLVMValueRef Val)


.. py:function:: LLVMIsAPtrToIntInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAPtrToIntInst(LLVMValueRef Val)


.. py:function:: LLVMIsASExtInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsASExtInst(LLVMValueRef Val)


.. py:function:: LLVMIsASIToFPInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsASIToFPInst(LLVMValueRef Val)


.. py:function:: LLVMIsATruncInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsATruncInst(LLVMValueRef Val)


.. py:function:: LLVMIsAUIToFPInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAUIToFPInst(LLVMValueRef Val)


.. py:function:: LLVMIsAZExtInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAZExtInst(LLVMValueRef Val)


.. py:function:: LLVMIsAExtractValueInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAExtractValueInst(LLVMValueRef Val)


.. py:function:: LLVMIsALoadInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsALoadInst(LLVMValueRef Val)


.. py:function:: LLVMIsAVAArgInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAVAArgInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFreezeInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFreezeInst(LLVMValueRef Val)


.. py:function:: LLVMIsAAtomicCmpXchgInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAAtomicCmpXchgInst(LLVMValueRef Val)


.. py:function:: LLVMIsAAtomicRMWInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAAtomicRMWInst(LLVMValueRef Val)


.. py:function:: LLVMIsAFenceInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAFenceInst(LLVMValueRef Val)


.. py:function:: LLVMIsABranchInst(Val)

   (No short description, might be part of a group.)

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsABranchInst(LLVMValueRef Val)


.. py:function:: LLVMIsAMDNode(Val)

   Convert value instances between types.

   Internally, an LLVMValueRef is "pinned" to a specific type. This
   series of functions allows you to cast an instance to a specific
   type.

   If the cast is not valid for the specified type, NULL is returned.

   See:
       llvm::dyn_cast_or_null<>

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMDNode(LLVMValueRef Val)


.. py:function:: LLVMIsAValueAsMetadata(Val)

   Convert value instances between types.

   Internally, an LLVMValueRef is "pinned" to a specific type. This
   series of functions allows you to cast an instance to a specific
   type.

   If the cast is not valid for the specified type, NULL is returned.

   See:
       llvm::dyn_cast_or_null<>

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAValueAsMetadata(LLVMValueRef Val)


.. py:function:: LLVMIsAMDString(Val)

   Convert value instances between types.

   Internally, an LLVMValueRef is "pinned" to a specific type. This
   series of functions allows you to cast an instance to a specific
   type.

   If the cast is not valid for the specified type, NULL is returned.

   See:
       llvm::dyn_cast_or_null<>

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsAMDString(LLVMValueRef Val)


.. py:function:: LLVMGetValueName(Val)

   Deprecated: Use LLVMGetValueName2 instead.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetValueName(LLVMValueRef Val)


.. py:function:: LLVMSetValueName(Val, Name)

   Deprecated: Use LLVMSetValueName2 instead.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetValueName(LLVMValueRef Val, const char * Name)


.. py:function:: LLVMGetFirstUse(Val)

   Obtain the first use of a value.

   Uses are obtained in an iterator fashion. First, call this function
   to obtain a reference to the first use. Then, call LLVMGetNextUse()
   on that instance and all subsequently obtained instances until
   LLVMGetNextUse() returns NULL.

   See:
       llvm::Value::use_begin()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMUseRef LLVMGetFirstUse(LLVMValueRef Val)


.. py:function:: LLVMGetNextUse(U)

   Obtain the next use of a value.

   This effectively advances the iterator. It returns NULL if you are on
   the final use and no more are available.

   Args:
       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMUseRef LLVMGetNextUse(LLVMUseRef U)


.. py:function:: LLVMGetUser(U)

   Obtain the user value for a user.

   The returned value corresponds to a llvm::User type.

   See:
       llvm::Use::getUser()

   Args:
       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetUser(LLVMUseRef U)


.. py:function:: LLVMGetUsedValue(U)

   Obtain the value this use corresponds to.

   See:
       llvm::Use::get().

   Args:
       U (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetUsedValue(LLVMUseRef U)


.. py:function:: LLVMGetOperand(Val, Index)

   Obtain an operand at a specific index in a llvm::User value.

   See:
       llvm::User::getOperand()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetOperand(LLVMValueRef Val, unsigned int Index)


.. py:function:: LLVMGetOperandUse(Val, Index)

   Obtain the use of an operand at a specific index in a llvm::User value.

   See:
       llvm::User::getOperandUse()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMUseRef LLVMGetOperandUse(LLVMValueRef Val, unsigned int Index)


.. py:function:: LLVMSetOperand(User, Index, Val)

   Set an operand at a specific index in a llvm::User value.

   See:
       llvm::User::setOperand()

   Args:
       User (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetOperand(LLVMValueRef User, unsigned int Index, LLVMValueRef Val)


.. py:function:: LLVMGetNumOperands(Val)

   Obtain the number of operands in a llvm::User value.

   See:
       llvm::User::getNumOperands()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMGetNumOperands(LLVMValueRef Val)


.. py:function:: LLVMConstNull(Ty)

   Obtain a constant value referring to the null instance of a type.

   See:
       llvm::Constant::getNullValue()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNull(LLVMTypeRef Ty)


.. py:function:: LLVMConstAllOnes(Ty)

   Obtain a constant value referring to the instance of a type
   consisting of all ones.

   This is only valid for integer types.

   See:
       llvm::Constant::getAllOnesValue()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty)


.. py:function:: LLVMGetUndef(Ty)

   Obtain a constant value referring to an undefined value of a type.

   See:
       llvm::UndefValue::get()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty)


.. py:function:: LLVMGetPoison(Ty)

   Obtain a constant value referring to a poison value of a type.

   See:
       llvm::PoisonValue::get()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPoison(LLVMTypeRef Ty)


.. py:function:: LLVMIsNull(Val)

   Determine whether a value instance is null.

   See:
       llvm::Constant::isNullValue()

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsNull(LLVMValueRef Val)


.. py:function:: LLVMConstPointerNull(Ty)

   Obtain a constant that is a constant pointer pointing to NULL for a
   specified type.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstPointerNull(LLVMTypeRef Ty)


.. py:function:: LLVMConstInt(IntTy, N, SignExtend)

   Obtain a constant value for an integer type.

   The returned value corresponds to a llvm::ConstantInt.

   See:
       llvm::ConstantInt::get()

   Args:
       IntTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Integer type to obtain value of.

       N (:py:obj:`~.int`):
           The value the returned instance should refer to.

       SignExtend (:py:obj:`~.int`):
           Whether to sign extend the produced value.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, unsigned long long N, LLVMBool SignExtend)


.. py:function:: LLVMConstIntOfArbitraryPrecision(IntTy, NumWords, Words)

   Obtain a constant value for an integer of arbitrary precision.

   See:
       llvm::ConstantInt::get()

   Args:
       IntTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumWords (:py:obj:`~.int`):
           (undocumented)

       Words (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstIntOfArbitraryPrecision(LLVMTypeRef IntTy, unsigned int NumWords, const uint64_t[] Words)


.. py:function:: LLVMConstIntOfString(IntTy, Text, Radix)

   Obtain a constant value for an integer parsed from a string.

   A similar API, LLVMConstIntOfStringAndSize is also available. If the
   string's length is available, it is preferred to call that function
   instead.

   See:
       llvm::ConstantInt::get()

   Args:
       IntTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Text (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Radix (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstIntOfString(LLVMTypeRef IntTy, const char * Text, uint8_t Radix)


.. py:function:: LLVMConstIntOfStringAndSize(IntTy, Text, SLen, Radix)

   Obtain a constant value for an integer parsed from a string with
   specified length.

   See:
       llvm::ConstantInt::get()

   Args:
       IntTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Text (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

       Radix (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstIntOfStringAndSize(LLVMTypeRef IntTy, const char * Text, unsigned int SLen, uint8_t Radix)


.. py:function:: LLVMConstByte(ByteTy, N)

   Obtain a constant value for a byte type.

   The returned value corresponds to a llvm::ConstantByte.

   See:
       llvm::ConstantByte::get()

   Args:
       ByteTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Byte type to obtain value of.

       N (:py:obj:`~.int`):
           The value the returned instance should refer to.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstByte(LLVMTypeRef ByteTy, unsigned long long N)


.. py:function:: LLVMConstByteOfArbitraryPrecision(ByteTy, NumWords, Words)

   Obtain a constant value for a byte of arbitrary precision.

   See:
       llvm::ConstantByte::get()

   Args:
       ByteTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumWords (:py:obj:`~.int`):
           (undocumented)

       Words (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstByteOfArbitraryPrecision(LLVMTypeRef ByteTy, unsigned int NumWords, const uint64_t[] Words)


.. py:function:: LLVMConstByteOfStringAndSize(ByteTy, Text, SLen, Radix)

   Obtain a constant value for a byte parsed from a string with specified
   length.

   See:
       llvm::ConstantByte::get()

   Args:
       ByteTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Text (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

       Radix (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstByteOfStringAndSize(LLVMTypeRef ByteTy, const char * Text, size_t SLen, uint8_t Radix)


.. py:function:: LLVMConstReal(RealTy, N)

   Obtain a constant value referring to a double floating point value.

   Args:
       RealTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       N (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N)


.. py:function:: LLVMConstRealOfString(RealTy, Text)

   Obtain a constant for a floating point value parsed from a string.

   A similar API, LLVMConstRealOfStringAndSize is also available. It
   should be used if the input string's length is known.

   Args:
       RealTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Text (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, const char * Text)


.. py:function:: LLVMConstRealOfStringAndSize(RealTy, Text, SLen)

   Obtain a constant for a floating point value parsed from a string.

   Args:
       RealTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Text (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstRealOfStringAndSize(LLVMTypeRef RealTy, const char * Text, unsigned int SLen)


.. py:function:: LLVMConstFPFromBits(Ty, N)

   Obtain a constant for a floating point value from array of 64 bit values.

   The length of the array N must be ceildiv(bits, 64), where bits is the
   scalar size in bits of the floating-point type.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       N (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstFPFromBits(LLVMTypeRef Ty, const uint64_t[] N)


.. py:function:: LLVMConstIntGetZExtValue(ConstantVal)

   Obtain the zero extended value for an integer constant value.

   See:
       llvm::ConstantInt::getZExtValue()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstIntGetSExtValue(ConstantVal)

   Obtain the sign extended value for an integer constant value.

   See:
       llvm::ConstantInt::getSExtValue()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       long long LLVMConstIntGetSExtValue(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstByteGetZExtValue(ConstantVal)

   Obtain the zero extended value for a byte constant value.

   See:
       llvm::ConstantByte::getZExtValue()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMConstByteGetZExtValue(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstByteGetSExtValue(ConstantVal)

   Obtain the sign extended value for a byte constant value.

   See:
       llvm::ConstantByte::getSExtValue()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       long long LLVMConstByteGetSExtValue(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstRealGetDouble(ConstantVal, losesInfo)

   Obtain the double value for an floating point constant value.

   losesInfo indicates if some precision was lost in the conversion.

   See:
       llvm::ConstantFP::getDoubleValue

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       losesInfo (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.float`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       double LLVMConstRealGetDouble(LLVMValueRef ConstantVal, LLVMBool * losesInfo)


.. py:function:: LLVMConstStringInContext(C, Str, Length, DontNullTerminate)

   Create a ConstantDataSequential and initialize it with a string.

   Deprecated:
       LLVMConstStringInContext is deprecated in favor of the API
       accurate LLVMConstStringInContext2

   See:
       llvm::ConstantDataArray::getString()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

       DontNullTerminate (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstStringInContext(LLVMContextRef C, const char * Str, unsigned int Length, LLVMBool DontNullTerminate)


.. py:function:: LLVMConstStringInContext2(C, Str, Length, DontNullTerminate)

   Create a ConstantDataSequential and initialize it with a string.

   See:
       llvm::ConstantDataArray::getString()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

       DontNullTerminate (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstStringInContext2(LLVMContextRef C, const char * Str, size_t Length, LLVMBool DontNullTerminate)


.. py:function:: LLVMConstString(Str, Length, DontNullTerminate)

   Create a ConstantDataSequential with string content in the global context.

   This is the same as LLVMConstStringInContext except it operates on the
   global context.

   See:
       :py:obj:`~.LLVMConstStringInContext`

   See:
       llvm::ConstantDataArray::getString()

   Args:
       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

       DontNullTerminate (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstString(const char * Str, unsigned int Length, LLVMBool DontNullTerminate)


.. py:function:: LLVMIsConstantString(c)

   Returns true if the specified constant is an array of i8.

   See:
       ConstantDataSequential::getAsString()

   Args:
       c (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsConstantString(LLVMValueRef c)


.. py:function:: LLVMGetAsString(c, Length)

   Get the given constant data sequential as a string.

   See:
       ConstantDataSequential::getAsString()

   Args:
       c (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetAsString(LLVMValueRef c, size_t * Length)


.. py:function:: LLVMGetRawDataValues(c, SizeInBytes)

   Get the raw, underlying bytes of the given constant data sequential.

   This is the same as LLVMGetAsString except it works for all constant data
   sequentials, not just i8 arrays.

   See:
       ConstantDataSequential::getRawDataValues()

   Args:
       c (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SizeInBytes (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetRawDataValues(LLVMValueRef c, size_t * SizeInBytes)


.. py:function:: LLVMConstStructInContext(C, ConstantVals, Count, Packed)

   Create an anonymous ConstantStruct with the specified values.

   See:
       llvm::ConstantStruct::getAnon()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

       Packed (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstStructInContext(LLVMContextRef C, LLVMValueRef * ConstantVals, unsigned int Count, LLVMBool Packed)


.. py:function:: LLVMConstStruct(ConstantVals, Count, Packed)

   Create a ConstantStruct in the global Context.

   This is the same as LLVMConstStructInContext except it operates on the
   global Context.

   See:
       :py:obj:`~.LLVMConstStructInContext`

   Args:
       ConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

       Packed (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstStruct(LLVMValueRef * ConstantVals, unsigned int Count, LLVMBool Packed)


.. py:function:: LLVMConstArray(ElementTy, ConstantVals, Length)

   Create a ConstantArray from values.

   Deprecated:
       LLVMConstArray is deprecated in favor of the API accurate
       LLVMConstArray2

   See:
       llvm::ConstantArray::get()

   Args:
       ElementTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy, LLVMValueRef * ConstantVals, unsigned int Length)


.. py:function:: LLVMConstArray2(ElementTy, ConstantVals, Length)

   Create a ConstantArray from values.

   See:
       llvm::ConstantArray::get()

   Args:
       ElementTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstArray2(LLVMTypeRef ElementTy, LLVMValueRef * ConstantVals, uint64_t Length)


.. py:function:: LLVMConstDataArray(ElementTy, Data, SizeInBytes)

   Create a ConstantDataArray from raw values.

   ElementTy must be one of i8, i16, i32, i64, half, bfloat, float, or double.
   Data points to a contiguous buffer of raw values in the host endianness. The
   element count is inferred from the element type and the data size in bytes.

   See:
       llvm::ConstantDataArray::getRaw()

   Args:
       ElementTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Data (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SizeInBytes (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstDataArray(LLVMTypeRef ElementTy, const char * Data, size_t SizeInBytes)


.. py:function:: LLVMConstNamedStruct(StructTy, ConstantVals, Count)

   Create a non-anonymous ConstantStruct from values.

   See:
       llvm::ConstantStruct::get()

   Args:
       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNamedStruct(LLVMTypeRef StructTy, LLVMValueRef * ConstantVals, unsigned int Count)


.. py:function:: LLVMGetAggregateElement(C, Idx)

   Get element of a constant aggregate (struct, array or vector) at the
   specified index.

   Returns null if the index is out of range, or it's not
   possible to determine the element (e.g., because the constant is a
   constant expression.)

   See:
       llvm::Constant::getAggregateElement()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetAggregateElement(LLVMValueRef C, unsigned int Idx)


.. py:function:: LLVMGetElementAsConstant(C, idx)

   Get an element at specified index as a constant.

   See:
       ConstantDataSequential::getElementAsConstant()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetElementAsConstant(LLVMValueRef C, unsigned int idx)


.. py:function:: LLVMConstVector(ScalarConstantVals, Size)

   Create a ConstantVector from values.

   See:
       llvm::ConstantVector::get()

   Args:
       ScalarConstantVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstVector(LLVMValueRef * ScalarConstantVals, unsigned int Size)


.. py:function:: LLVMConstantPtrAuth(Ptr, Key, Disc, AddrDisc)

   Create a ConstantPtrAuth constant with the given values.

   See:
       llvm::ConstantPtrAuth::get()

   Args:
       Ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Key (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Disc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AddrDisc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstantPtrAuth(LLVMValueRef Ptr, LLVMValueRef Key, LLVMValueRef Disc, LLVMValueRef AddrDisc)


.. py:function:: LLVMGetConstOpcode(ConstantVal)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMOpcode`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOpcode LLVMGetConstOpcode(LLVMValueRef ConstantVal)


.. py:function:: LLVMAlignOf(Ty)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAlignOf(LLVMTypeRef Ty)


.. py:function:: LLVMSizeOf(Ty)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty)


.. py:function:: LLVMConstNeg(ConstantVal)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstNSWNeg(ConstantVal)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNSWNeg(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstNUWNeg(ConstantVal)

   (No short description, might be part of a group.)

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNUWNeg(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstNot(ConstantVal)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal)


.. py:function:: LLVMConstAdd(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstNSWAdd(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNSWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstNUWAdd(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNUWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstSub(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstNSWSub(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNSWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstNUWSub(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstNUWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstXor(LHSConstant, RHSConstant)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       LHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHSConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant)


.. py:function:: LLVMConstGEP2(Ty, ConstantVal, ConstantIndices, NumIndices)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantIndices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstGEP2(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef * ConstantIndices, unsigned int NumIndices)


.. py:function:: LLVMConstInBoundsGEP2(Ty, ConstantVal, ConstantIndices, NumIndices)

   Functions in this group correspond to APIs on llvm::ConstantExpr.

   See:
       llvm::ConstantExpr.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantIndices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstInBoundsGEP2(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef * ConstantIndices, unsigned int NumIndices)


.. py:function:: LLVMConstGEPWithNoWrapFlags(Ty, ConstantVal, ConstantIndices, NumIndices, NoWrapFlags)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantIndices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

       NoWrapFlags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstGEPWithNoWrapFlags(LLVMTypeRef Ty, LLVMValueRef ConstantVal, LLVMValueRef * ConstantIndices, unsigned int NumIndices, LLVMGEPNoWrapFlags NoWrapFlags)


.. py:function:: LLVMConstTrunc(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstPtrToInt(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstIntToPtr(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstBitCast(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstAddrSpaceCast(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstAddrSpaceCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstTruncOrBitCast(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstTruncOrBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstPointerCast(ConstantVal, ToType)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ToType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstPointerCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType)


.. py:function:: LLVMConstExtractElement(VectorConstant, IndexConstant)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       VectorConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IndexConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant, LLVMValueRef IndexConstant)


.. py:function:: LLVMConstInsertElement(VectorConstant, ElementValueConstant, IndexConstant)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       VectorConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElementValueConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IndexConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant, LLVMValueRef ElementValueConstant, LLVMValueRef IndexConstant)


.. py:function:: LLVMConstShuffleVector(VectorAConstant, VectorBConstant, MaskConstant)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       VectorAConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       VectorBConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MaskConstant (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant, LLVMValueRef VectorBConstant, LLVMValueRef MaskConstant)


.. py:function:: LLVMBlockAddress(F, BB)

   Creates a constant GetElementPtr expression.

   Similar to LLVMConstGEP2, but
   allows specifying the no-wrap flags.

   See:
       llvm::ConstantExpr::getGetElementPtr()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBlockAddress(LLVMValueRef F, LLVMBasicBlockRef BB)


.. py:function:: LLVMGetBlockAddressFunction(BlockAddr)

   Gets the function associated with a given BlockAddress constant value.

   Args:
       BlockAddr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetBlockAddressFunction(LLVMValueRef BlockAddr)


.. py:function:: LLVMGetBlockAddressBasicBlock(BlockAddr)

   Gets the basic block associated with a given BlockAddress constant value.

   Args:
       BlockAddr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetBlockAddressBasicBlock(LLVMValueRef BlockAddr)


.. py:function:: LLVMConstInlineAsm(Ty, AsmString, Constraints, HasSideEffects, IsAlignStack)

   Deprecated: Use LLVMGetInlineAsm instead.

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AsmString (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Constraints (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       HasSideEffects (:py:obj:`~.int`):
           (undocumented)

       IsAlignStack (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMConstInlineAsm(LLVMTypeRef Ty, const char * AsmString, const char * Constraints, LLVMBool HasSideEffects, LLVMBool IsAlignStack)


.. py:function:: LLVMGetGlobalParent(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global)


.. py:function:: LLVMIsDeclaration(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsDeclaration(LLVMValueRef Global)


.. py:function:: LLVMGetLinkage(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMLinkage`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMLinkage LLVMGetLinkage(LLVMValueRef Global)


.. py:function:: LLVMSetLinkage(Global, Linkage)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Linkage (:py:obj:`~.LLVMLinkage`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage)


.. py:function:: LLVMGetSection(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetSection(LLVMValueRef Global)


.. py:function:: LLVMSetSection(Global, Section)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Section (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetSection(LLVMValueRef Global, const char * Section)


.. py:function:: LLVMGetVisibility(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMVisibility`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMVisibility LLVMGetVisibility(LLVMValueRef Global)


.. py:function:: LLVMSetVisibility(Global, Viz)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Viz (:py:obj:`~.LLVMVisibility`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz)


.. py:function:: LLVMGetDLLStorageClass(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMDLLStorageClass`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDLLStorageClass LLVMGetDLLStorageClass(LLVMValueRef Global)


.. py:function:: LLVMSetDLLStorageClass(Global, Class)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Class (:py:obj:`~.LLVMDLLStorageClass`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetDLLStorageClass(LLVMValueRef Global, LLVMDLLStorageClass Class)


.. py:function:: LLVMGetUnnamedAddress(Global)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMUnnamedAddr`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMUnnamedAddr LLVMGetUnnamedAddress(LLVMValueRef Global)


.. py:function:: LLVMSetUnnamedAddress(Global, UnnamedAddr)

   This group contains functions that operate on global values.

   Functions in
   this group relate to functions in the llvm::GlobalValue class tree.

   See:
       llvm::GlobalValue

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       UnnamedAddr (:py:obj:`~.LLVMUnnamedAddr`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetUnnamedAddress(LLVMValueRef Global, LLVMUnnamedAddr UnnamedAddr)


.. py:function:: LLVMGlobalGetValueType(Global)

   Returns the "value type" of a global value.

   This differs from the formal
   type of a global value which is always a pointer type.

   See:
       llvm::GlobalValue::getValueType()

   See:
       llvm::Function::getFunctionType()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGlobalGetValueType(LLVMValueRef Global)


.. py:function:: LLVMHasUnnamedAddr(Global)

   Deprecated: Use LLVMGetUnnamedAddress instead.

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMHasUnnamedAddr(LLVMValueRef Global)


.. py:function:: LLVMSetUnnamedAddr(Global, HasUnnamedAddr)

   Deprecated: Use LLVMSetUnnamedAddress instead.

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       HasUnnamedAddr (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetUnnamedAddr(LLVMValueRef Global, LLVMBool HasUnnamedAddr)


.. py:function:: LLVMGetAlignment(V)

   Obtain the preferred alignment of the value.

   See:
       llvm::AllocaInst::getAlignment()

   See:
       llvm::LoadInst::getAlignment()

   See:
       llvm::StoreInst::getAlignment()

   See:
       llvm::AtomicRMWInst::setAlignment()

   See:
       llvm::AtomicCmpXchgInst::setAlignment()

   See:
       llvm::GlobalValue::getAlignment()

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetAlignment(LLVMValueRef V)


.. py:function:: LLVMSetAlignment(V, Bytes)

   Set the preferred alignment of the value.

   See:
       llvm::AllocaInst::setAlignment()

   See:
       llvm::LoadInst::setAlignment()

   See:
       llvm::StoreInst::setAlignment()

   See:
       llvm::AtomicRMWInst::setAlignment()

   See:
       llvm::AtomicCmpXchgInst::setAlignment()

   See:
       llvm::GlobalValue::setAlignment()

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Bytes (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetAlignment(LLVMValueRef V, unsigned int Bytes)


.. py:function:: LLVMGlobalSetMetadata(Global, Kind, MD)

   Sets a metadata attachment, erasing the existing metadata attachment if
   it already exists for the given kind.

   See:
       llvm::GlobalObject::setMetadata()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Kind (:py:obj:`~.int`):
           (undocumented)

       MD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGlobalSetMetadata(LLVMValueRef Global, unsigned int Kind, LLVMMetadataRef MD)


.. py:function:: LLVMGlobalAddMetadata(Global, Kind, MD)

   Adds a metadata attachment.

   See:
       llvm::GlobalObject::addMetadata()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Kind (:py:obj:`~.int`):
           (undocumented)

       MD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGlobalAddMetadata(LLVMValueRef Global, unsigned int Kind, LLVMMetadataRef MD)


.. py:function:: LLVMGlobalEraseMetadata(Global, Kind)

   Erases a metadata attachment of the given kind if it exists.

   See:
       llvm::GlobalObject::eraseMetadata()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Kind (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGlobalEraseMetadata(LLVMValueRef Global, unsigned int Kind)


.. py:function:: LLVMGlobalClearMetadata(Global)

   Removes all metadata attachments from this value.

   See:
       llvm::GlobalObject::clearMetadata()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGlobalClearMetadata(LLVMValueRef Global)


.. py:function:: LLVMGlobalAddDebugInfo(Global, GVE)

   Add debuginfo metadata to this global.

   See:
       llvm::GlobalVariable::addDebugInfo()

   Args:
       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       GVE (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGlobalAddDebugInfo(LLVMValueRef Global, LLVMMetadataRef GVE)


.. py:function:: LLVMGlobalCopyAllMetadata(Value, NumEntries)

   Retrieves an array of metadata entries representing the metadata attached to
   this value.

   The caller is responsible for freeing this array by calling
   ``LLVMDisposeValueMetadataEntries.``

   See:
       llvm::GlobalObject::getAllMetadata()

   Args:
       Value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumEntries (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueMetadataEntry * LLVMGlobalCopyAllMetadata(LLVMValueRef Value, size_t * NumEntries)


.. py:function:: LLVMDisposeValueMetadataEntries(Entries)

   Destroys value metadata entries.

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeValueMetadataEntries(LLVMValueMetadataEntry * Entries)


.. py:function:: LLVMValueMetadataEntriesGetKind(Entries, Index)

   Returns the kind of a value metadata entry at a specific index.

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMValueMetadataEntriesGetKind(LLVMValueMetadataEntry * Entries, unsigned int Index)


.. py:function:: LLVMValueMetadataEntriesGetMetadata(Entries, Index)

   Returns the underlying metadata node of a value metadata entry at a
   specific index.

   Args:
       Entries (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMValueMetadataEntriesGetMetadata(LLVMValueMetadataEntry * Entries, unsigned int Index)


.. py:function:: LLVMAddGlobal(M, Ty, Name)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char * Name)


.. py:function:: LLVMAddGlobalInAddressSpace(M, Ty, Name, AddressSpace)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       AddressSpace (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAddGlobalInAddressSpace(LLVMModuleRef M, LLVMTypeRef Ty, const char * Name, unsigned int AddressSpace)


.. py:function:: LLVMGetNamedGlobal(M, Name)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, const char * Name)


.. py:function:: LLVMGetNamedGlobalWithLength(M, Name, Length)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedGlobalWithLength(LLVMModuleRef M, const char * Name, size_t Length)


.. py:function:: LLVMGetFirstGlobal(M)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M)


.. py:function:: LLVMGetLastGlobal(M)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M)


.. py:function:: LLVMGetNextGlobal(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar)


.. py:function:: LLVMGetPreviousGlobal(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar)


.. py:function:: LLVMDeleteGlobal(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDeleteGlobal(LLVMValueRef GlobalVar)


.. py:function:: LLVMGetInitializer(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar)


.. py:function:: LLVMSetInitializer(GlobalVar, ConstantVal)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal)


.. py:function:: LLVMIsThreadLocal(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsThreadLocal(LLVMValueRef GlobalVar)


.. py:function:: LLVMSetThreadLocal(GlobalVar, IsThreadLocal)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsThreadLocal (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetThreadLocal(LLVMValueRef GlobalVar, LLVMBool IsThreadLocal)


.. py:function:: LLVMIsGlobalConstant(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsGlobalConstant(LLVMValueRef GlobalVar)


.. py:function:: LLVMSetGlobalConstant(GlobalVar, IsConstant)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsConstant (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, LLVMBool IsConstant)


.. py:function:: LLVMGetThreadLocalMode(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMThreadLocalMode`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMThreadLocalMode LLVMGetThreadLocalMode(LLVMValueRef GlobalVar)


.. py:function:: LLVMSetThreadLocalMode(GlobalVar, Mode)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Mode (:py:obj:`~.LLVMThreadLocalMode`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetThreadLocalMode(LLVMValueRef GlobalVar, LLVMThreadLocalMode Mode)


.. py:function:: LLVMIsExternallyInitialized(GlobalVar)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsExternallyInitialized(LLVMValueRef GlobalVar)


.. py:function:: LLVMSetExternallyInitialized(GlobalVar, IsExtInit)

   This group contains functions that operate on global variable values.

   See:
       llvm::GlobalVariable

   Args:
       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsExtInit (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetExternallyInitialized(LLVMValueRef GlobalVar, LLVMBool IsExtInit)


.. py:function:: LLVMAddAlias2(M, ValueTy, AddrSpace, Aliasee, Name)

   Add a GlobalAlias with the given value type, address space and aliasee.

   See:
       llvm::GlobalAlias::create()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ValueTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AddrSpace (:py:obj:`~.int`):
           (undocumented)

       Aliasee (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAddAlias2(LLVMModuleRef M, LLVMTypeRef ValueTy, unsigned int AddrSpace, LLVMValueRef Aliasee, const char * Name)


.. py:function:: LLVMGetNamedGlobalAlias(M, Name, NameLen)

   Obtain a GlobalAlias value from a Module by its name.

   The returned value corresponds to a llvm::GlobalAlias value.

   See:
       llvm::Module::getNamedAlias()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedGlobalAlias(LLVMModuleRef M, const char * Name, size_t NameLen)


.. py:function:: LLVMGetFirstGlobalAlias(M)

   Obtain an iterator to the first GlobalAlias in a Module.

   See:
       llvm::Module::alias_begin()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstGlobalAlias(LLVMModuleRef M)


.. py:function:: LLVMGetLastGlobalAlias(M)

   Obtain an iterator to the last GlobalAlias in a Module.

   See:
       llvm::Module::alias_end()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastGlobalAlias(LLVMModuleRef M)


.. py:function:: LLVMGetNextGlobalAlias(GA)

   Advance a GlobalAlias iterator to the next GlobalAlias.

   Returns NULL if the iterator was already at the end and there are no more
   global aliases.

   Args:
       GA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextGlobalAlias(LLVMValueRef GA)


.. py:function:: LLVMGetPreviousGlobalAlias(GA)

   Decrement a GlobalAlias iterator to the previous GlobalAlias.

   Returns NULL if the iterator was already at the beginning and there are
   no previous global aliases.

   Args:
       GA (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousGlobalAlias(LLVMValueRef GA)


.. py:function:: LLVMAliasGetAliasee(Alias)

   Retrieve the target value of an alias.

   Args:
       Alias (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAliasGetAliasee(LLVMValueRef Alias)


.. py:function:: LLVMAliasSetAliasee(Alias, Aliasee)

   Set the target value of an alias.

   Args:
       Alias (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Aliasee (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAliasSetAliasee(LLVMValueRef Alias, LLVMValueRef Aliasee)


.. py:function:: LLVMDeleteFunction(Fn)

   Remove a function from its containing module and deletes it.

   See:
       llvm::Function::eraseFromParent()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDeleteFunction(LLVMValueRef Fn)


.. py:function:: LLVMHasPersonalityFn(Fn)

   Check whether the given function has a personality function.

   See:
       llvm::Function::hasPersonalityFn()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMHasPersonalityFn(LLVMValueRef Fn)


.. py:function:: LLVMGetPersonalityFn(Fn)

   Obtain the personality function attached to the function.

   See:
       llvm::Function::getPersonalityFn()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPersonalityFn(LLVMValueRef Fn)


.. py:function:: LLVMSetPersonalityFn(Fn, PersonalityFn)

   Set the personality function attached to the function.

   See:
       llvm::Function::setPersonalityFn()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       PersonalityFn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetPersonalityFn(LLVMValueRef Fn, LLVMValueRef PersonalityFn)


.. py:function:: LLVMLookupIntrinsicID(Name, NameLen)

   Obtain the intrinsic ID number which matches the given function name.

   See:
       llvm::Intrinsic::lookupIntrinsicID()

   Args:
       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMLookupIntrinsicID(const char * Name, size_t NameLen)


.. py:function:: LLVMGetIntrinsicID(Fn)

   Obtain the ID number from a function instance.

   See:
       llvm::Function::getIntrinsicID()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetIntrinsicID(LLVMValueRef Fn)


.. py:function:: LLVMGetIntrinsicDeclaration(Mod, ID, OverloadTypes, OverloadCount)

   Get or insert the declaration of an intrinsic.

   For overloaded intrinsics,
   overload types must be provided to uniquely identify an overload.

   See:
       llvm::Intrinsic::getOrInsertDeclaration()

   Args:
       Mod (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ID (:py:obj:`~.int`):
           (undocumented)

       OverloadTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OverloadCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetIntrinsicDeclaration(LLVMModuleRef Mod, unsigned int ID, LLVMTypeRef * OverloadTypes, size_t OverloadCount)


.. py:function:: LLVMIntrinsicGetType(Ctx, ID, OverloadTypes, OverloadCount)

   Retrieves the type of an intrinsic.

   For overloaded intrinsics, overload
   types must be provided to uniquely identify an overload.

   See:
       llvm::Intrinsic::getType()

   Args:
       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ID (:py:obj:`~.int`):
           (undocumented)

       OverloadTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OverloadCount (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntrinsicGetType(LLVMContextRef Ctx, unsigned int ID, LLVMTypeRef * OverloadTypes, size_t OverloadCount)


.. py:function:: LLVMIntrinsicGetName(ID, NameLength)

   Retrieves the name of an intrinsic.

   See:
       llvm::Intrinsic::getName()

   Args:
       ID (:py:obj:`~.int`):
           (undocumented)

       NameLength (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMIntrinsicGetName(unsigned int ID, size_t * NameLength)


.. py:function:: LLVMIntrinsicCopyOverloadedName(ID, OverloadTypes, OverloadCount, NameLength)

   Deprecated: Use LLVMIntrinsicCopyOverloadedName2 instead.

   Args:
       ID (:py:obj:`~.int`):
           (undocumented)

       OverloadTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OverloadCount (:py:obj:`~.int`):
           (undocumented)

       NameLength (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMIntrinsicCopyOverloadedName(unsigned int ID, LLVMTypeRef * OverloadTypes, size_t OverloadCount, size_t * NameLength)


.. py:function:: LLVMIntrinsicCopyOverloadedName2(Mod, ID, OverloadTypes, OverloadCount, NameLength)

   Copies the name of an overloaded intrinsic identified by a given list of
   overload types.

   Unlike LLVMIntrinsicGetName, the caller is responsible for freeing the
   returned string.

   This version also supports unnamed types.

   See:
       llvm::Intrinsic::getName()

   Args:
       Mod (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ID (:py:obj:`~.int`):
           (undocumented)

       OverloadTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OverloadCount (:py:obj:`~.int`):
           (undocumented)

       NameLength (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMIntrinsicCopyOverloadedName2(LLVMModuleRef Mod, unsigned int ID, LLVMTypeRef * OverloadTypes, size_t OverloadCount, size_t * NameLength)


.. py:function:: LLVMIntrinsicIsOverloaded(ID)

   Obtain if the intrinsic identified by the given ID is overloaded.

   See:
       llvm::Intrinsic::isOverloaded()

   Args:
       ID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIntrinsicIsOverloaded(unsigned int ID)


.. py:function:: LLVMGetFunctionCallConv(Fn)

   Obtain the calling function of a function.

   The returned value corresponds to the LLVMCallConv enumeration.

   See:
       llvm::Function::getCallingConv()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetFunctionCallConv(LLVMValueRef Fn)


.. py:function:: LLVMSetFunctionCallConv(Fn, CC)

   Set the calling convention of a function.

   See:
       llvm::Function::setCallingConv()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Function to operate on

       CC (:py:obj:`~.int`):
           LLVMCallConv to set calling convention to

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned int CC)


.. py:function:: LLVMGetGC(Fn)

   Obtain the name of the garbage collector to use during code
   generation.

   See:
       llvm::Function::getGC()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetGC(LLVMValueRef Fn)


.. py:function:: LLVMSetGC(Fn, Name)

   Define the garbage collector to use during code generation.

   See:
       llvm::Function::setGC()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetGC(LLVMValueRef Fn, const char * Name)


.. py:function:: LLVMGetPrefixData(Fn)

   Gets the prefix data associated with a function.

   Only valid on functions, and
   only if LLVMHasPrefixData returns true.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prefix`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPrefixData(LLVMValueRef Fn)


.. py:function:: LLVMHasPrefixData(Fn)

   Check if a given function has prefix data.

   Only valid on functions.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prefix`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMHasPrefixData(LLVMValueRef Fn)


.. py:function:: LLVMSetPrefixData(Fn, prefixData)

   Sets the prefix data for the function.

   Only valid on functions.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prefix`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       prefixData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetPrefixData(LLVMValueRef Fn, LLVMValueRef prefixData)


.. py:function:: LLVMGetPrologueData(Fn)

   Gets the prologue data associated with a function.

   Only valid on functions,
   and only if LLVMHasPrologueData returns true.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prologue`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPrologueData(LLVMValueRef Fn)


.. py:function:: LLVMHasPrologueData(Fn)

   Check if a given function has prologue data.

   Only valid on functions.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prologue`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMHasPrologueData(LLVMValueRef Fn)


.. py:function:: LLVMSetPrologueData(Fn, prologueData)

   Sets the prologue data for the function.

   Only valid on functions.
   See https://llvm.org/docs/LangRef.html:py:obj:`~.prologue`-data

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       prologueData (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetPrologueData(LLVMValueRef Fn, LLVMValueRef prologueData)


.. py:function:: LLVMAddAttributeAtIndex(F, Idx, A)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, LLVMAttributeRef A)


.. py:function:: LLVMGetAttributeCountAtIndex(F, Idx)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetAttributeCountAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx)


.. py:function:: LLVMGetAttributesAtIndex(F, Idx, Attrs)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       Attrs (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetAttributesAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, LLVMAttributeRef * Attrs)


.. py:function:: LLVMGetEnumAttributeAtIndex(F, Idx, KindID)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMGetEnumAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, unsigned int KindID)


.. py:function:: LLVMGetStringAttributeAtIndex(F, Idx, K, KLen)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       K (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMGetStringAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, const char * K, unsigned int KLen)


.. py:function:: LLVMRemoveEnumAttributeAtIndex(F, Idx, KindID)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveEnumAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, unsigned int KindID)


.. py:function:: LLVMRemoveStringAttributeAtIndex(F, Idx, K, KLen)

   Add an attribute to a function.

   See:
       llvm::Function::addAttribute()

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       K (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KLen (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveStringAttributeAtIndex(LLVMValueRef F, LLVMAttributeIndex Idx, const char * K, unsigned int KLen)


.. py:function:: LLVMAddTargetDependentFunctionAttr(Fn, A, V)

   Add a target-dependent attribute to a function

   See:
       llvm::AttrBuilder::addAttribute()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddTargetDependentFunctionAttr(LLVMValueRef Fn, const char * A, const char * V)


.. py:function:: LLVMCountParams(Fn)

   Obtain the number of parameters in a function.

   See:
       llvm::Function::arg_size()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCountParams(LLVMValueRef Fn)


.. py:function:: LLVMGetParams(Fn, Params)

   Obtain the parameters in a function.

   The takes a pointer to a pre-allocated array of LLVMValueRef that is
   at least LLVMCountParams() long. This array will be filled with
   LLVMValueRef instances which correspond to the parameters the
   function receives. Each LLVMValueRef corresponds to a llvm::Argument
   instance.

   See:
       llvm::Function::arg_begin()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Params (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetParams(LLVMValueRef Fn, LLVMValueRef * Params)


.. py:function:: LLVMGetParam(Fn, Index)

   Obtain the parameter at the specified index.

   Parameters are indexed from 0.

   See:
       llvm::Function::arg_begin()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetParam(LLVMValueRef Fn, unsigned int Index)


.. py:function:: LLVMGetParamParent(Inst)

   Obtain the function to which this argument belongs.

   Unlike other functions in this group, this one takes an LLVMValueRef
   that corresponds to a llvm::Attribute.

   The returned LLVMValueRef is the llvm::Function to which this
   argument belongs.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetParamParent(LLVMValueRef Inst)


.. py:function:: LLVMGetFirstParam(Fn)

   Obtain the first parameter to a function.

   See:
       llvm::Function::arg_begin()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn)


.. py:function:: LLVMGetLastParam(Fn)

   Obtain the last parameter to a function.

   See:
       llvm::Function::arg_end()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn)


.. py:function:: LLVMGetNextParam(Arg)

   Obtain the next parameter to a function.

   This takes an LLVMValueRef obtained from LLVMGetFirstParam() (which is
   actually a wrapped iterator) and obtains the next parameter from the
   underlying iterator.

   Args:
       Arg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg)


.. py:function:: LLVMGetPreviousParam(Arg)

   Obtain the previous parameter to a function.

   This is the opposite of LLVMGetNextParam().

   Args:
       Arg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg)


.. py:function:: LLVMSetParamAlignment(Arg, Align)

   Set the alignment for a function parameter.

   See:
       llvm::Argument::addAttr()

   See:
       llvm::AttrBuilder::addAlignmentAttr()

   Args:
       Arg (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Align (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetParamAlignment(LLVMValueRef Arg, unsigned int Align)


.. py:function:: LLVMAddGlobalIFunc(M, Name, NameLen, Ty, AddrSpace, Resolver)

   Add a global indirect function to a module under a specified name.

   See:
       llvm::GlobalIFunc::create()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AddrSpace (:py:obj:`~.int`):
           (undocumented)

       Resolver (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMAddGlobalIFunc(LLVMModuleRef M, const char * Name, size_t NameLen, LLVMTypeRef Ty, unsigned int AddrSpace, LLVMValueRef Resolver)


.. py:function:: LLVMGetNamedGlobalIFunc(M, Name, NameLen)

   Obtain a GlobalIFunc value from a Module by its name.

   The returned value corresponds to a llvm::GlobalIFunc value.

   See:
       llvm::Module::getNamedIFunc()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNamedGlobalIFunc(LLVMModuleRef M, const char * Name, size_t NameLen)


.. py:function:: LLVMGetFirstGlobalIFunc(M)

   Obtain an iterator to the first GlobalIFunc in a Module.

   See:
       llvm::Module::ifunc_begin()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstGlobalIFunc(LLVMModuleRef M)


.. py:function:: LLVMGetLastGlobalIFunc(M)

   Obtain an iterator to the last GlobalIFunc in a Module.

   See:
       llvm::Module::ifunc_end()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastGlobalIFunc(LLVMModuleRef M)


.. py:function:: LLVMGetNextGlobalIFunc(IFunc)

   Advance a GlobalIFunc iterator to the next GlobalIFunc.

   Returns NULL if the iterator was already at the end and there are no more
   global aliases.

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextGlobalIFunc(LLVMValueRef IFunc)


.. py:function:: LLVMGetPreviousGlobalIFunc(IFunc)

   Decrement a GlobalIFunc iterator to the previous GlobalIFunc.

   Returns NULL if the iterator was already at the beginning and there are
   no previous global aliases.

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousGlobalIFunc(LLVMValueRef IFunc)


.. py:function:: LLVMGetGlobalIFuncResolver(IFunc)

   Retrieves the resolver function associated with this indirect function, or
   NULL if it doesn't not exist.

   See:
       llvm::GlobalIFunc::getResolver()

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetGlobalIFuncResolver(LLVMValueRef IFunc)


.. py:function:: LLVMSetGlobalIFuncResolver(IFunc, Resolver)

   Sets the resolver function associated with this indirect function.

   See:
       llvm::GlobalIFunc::setResolver()

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Resolver (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetGlobalIFuncResolver(LLVMValueRef IFunc, LLVMValueRef Resolver)


.. py:function:: LLVMEraseGlobalIFunc(IFunc)

   Remove a global indirect function from its parent module and delete it.

   See:
       llvm::GlobalIFunc::eraseFromParent()

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMEraseGlobalIFunc(LLVMValueRef IFunc)


.. py:function:: LLVMRemoveGlobalIFunc(IFunc)

   Remove a global indirect function from its parent module.

   This unlinks the global indirect function from its containing module but
   keeps it alive.

   See:
       llvm::GlobalIFunc::removeFromParent()

   Args:
       IFunc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveGlobalIFunc(LLVMValueRef IFunc)


.. py:function:: LLVMMDStringInContext2(C, Str, SLen)

   Create an MDString value from a given string value.

   The MDString value does not take ownership of the given string, it remains
   the responsibility of the caller to free it.

   See:
       llvm::MDString::get()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMMDStringInContext2(LLVMContextRef C, const char * Str, size_t SLen)


.. py:function:: LLVMMDNodeInContext2(C, MDs, Count)

   Create an MDNode value with the given array of operands.

   See:
       llvm::MDNode::get()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MDs (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMMDNodeInContext2(LLVMContextRef C, LLVMMetadataRef * MDs, size_t Count)


.. py:function:: LLVMMetadataAsValue(C, MD)

   Obtain a Metadata as a Value.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMMetadataAsValue(LLVMContextRef C, LLVMMetadataRef MD)


.. py:function:: LLVMValueAsMetadata(Val)

   Obtain a Value as a Metadata.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMValueAsMetadata(LLVMValueRef Val)


.. py:function:: LLVMGetMDString(V, Length)

   Obtain the underlying string from a MDString value.

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Instance to obtain string from.

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           Memory address which will hold length of returned string.

   Returns:
       :py:obj:`~.bytes`: String data in MDString.

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetMDString(LLVMValueRef V, unsigned int * Length)


.. py:function:: LLVMGetMDNodeNumOperands(V)

   Obtain the number of operands from an MDNode value.

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           MDNode to get number of operands from.

   Returns:
       :py:obj:`~.int`: Number of operands of the MDNode.

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetMDNodeNumOperands(LLVMValueRef V)


.. py:function:: LLVMGetMDNodeOperands(V, Dest)

   Obtain the given MDNode's operands.

   The passed LLVMValueRef pointer should point to enough memory to hold all of
   the operands of the given MDNode (see LLVMGetMDNodeNumOperands) as
   LLVMValueRefs. This memory will be populated with the LLVMValueRefs of the
   MDNode's operands.

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           MDNode to get the operands from.

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Destination array for operands.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetMDNodeOperands(LLVMValueRef V, LLVMValueRef * Dest)


.. py:function:: LLVMReplaceMDNodeOperandWith(V, Index, Replacement)

   Replace an operand at a specific index in a llvm::MDNode value.

   See:
       llvm::MDNode::replaceOperandWith()

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

       Replacement (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMReplaceMDNodeOperandWith(LLVMValueRef V, unsigned int Index, LLVMMetadataRef Replacement)


.. py:function:: LLVMMDStringInContext(C, Str, SLen)

   Deprecated: Use LLVMMDStringInContext2 instead.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMMDStringInContext(LLVMContextRef C, const char * Str, unsigned int SLen)


.. py:function:: LLVMMDString(Str, SLen)

   Deprecated: Use LLVMMDStringInContext2 instead.

   Args:
       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMMDString(const char * Str, unsigned int SLen)


.. py:function:: LLVMMDNodeInContext(C, Vals, Count)

   Deprecated: Use LLVMMDNodeInContext2 instead.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Vals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMMDNodeInContext(LLVMContextRef C, LLVMValueRef * Vals, unsigned int Count)


.. py:function:: LLVMMDNode(Vals, Count)

   Deprecated: Use LLVMMDNodeInContext2 instead.

   Args:
       Vals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMMDNode(LLVMValueRef * Vals, unsigned int Count)


.. py:function:: LLVMCreateOperandBundle(Tag, TagLen, Args, NumArgs)

   Create a new operand bundle.

   Every invocation should be paired with LLVMDisposeOperandBundle() or memory
   will be leaked.

   Args:
       Tag (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Tag name of the operand bundle

       TagLen (:py:obj:`~.int`):
           Length of Tag

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Memory address of an array of bundle operands

       NumArgs (:py:obj:`~.int`):
           Length of Args

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOperandBundleRef LLVMCreateOperandBundle(const char * Tag, size_t TagLen, LLVMValueRef * Args, unsigned int NumArgs)


.. py:function:: LLVMDisposeOperandBundle(Bundle)

   Destroy an operand bundle.

   This must be called for every created operand bundle or memory will be
   leaked.

   Args:
       Bundle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeOperandBundle(LLVMOperandBundleRef Bundle)


.. py:function:: LLVMGetOperandBundleTag(Bundle, Len)

   Obtain the tag of an operand bundle as a string.

   See:
       OperandBundleDef::getTag()

   Args:
       Bundle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Operand bundle to obtain tag of.

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           Out parameter which holds the length of the returned string.

   Returns:
       :py:obj:`~.bytes`: The tag name of Bundle.

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetOperandBundleTag(LLVMOperandBundleRef Bundle, size_t * Len)


.. py:function:: LLVMGetNumOperandBundleArgs(Bundle)

   Obtain the number of operands for an operand bundle.

   See:
       OperandBundleDef::input_size()

   Args:
       Bundle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Operand bundle to obtain operand count of.

   Returns:
       :py:obj:`~.int`: The number of operands.

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumOperandBundleArgs(LLVMOperandBundleRef Bundle)


.. py:function:: LLVMGetOperandBundleArgAtIndex(Bundle, Index)

   Obtain the operand for an operand bundle at the given index.

   Args:
       Bundle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Operand bundle to obtain operand of.

       Index (:py:obj:`~.int`):
           An operand index, must be less than
           LLVMGetNumOperandBundleArgs().

   Returns:
       :py:obj:`~.None`: The operand.

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetOperandBundleArgAtIndex(LLVMOperandBundleRef Bundle, unsigned int Index)


.. py:function:: LLVMBasicBlockAsValue(BB)

   Convert a basic block instance to a value type.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef BB)


.. py:function:: LLVMValueIsBasicBlock(Val)

   Determine whether an LLVMValueRef is itself a basic block.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMValueIsBasicBlock(LLVMValueRef Val)


.. py:function:: LLVMValueAsBasicBlock(Val)

   Convert an LLVMValueRef to an LLVMBasicBlockRef instance.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val)


.. py:function:: LLVMGetBasicBlockName(BB)

   Obtain the string name of a basic block.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetBasicBlockName(LLVMBasicBlockRef BB)


.. py:function:: LLVMGetBasicBlockParent(BB)

   Obtain the function to which a basic block belongs.

   See:
       llvm::BasicBlock::getParent()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB)


.. py:function:: LLVMGetBasicBlockTerminator(BB)

   Obtain the terminator instruction for a basic block.

   If the basic block does not have a terminator (it is not well-formed
   if it doesn't), then NULL is returned.

   The returned LLVMValueRef corresponds to an llvm::Instruction.

   See:
       llvm::BasicBlock::getTerminator()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef BB)


.. py:function:: LLVMCountBasicBlocks(Fn)

   Obtain the number of basic blocks in a function.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Function value to operate on.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCountBasicBlocks(LLVMValueRef Fn)


.. py:function:: LLVMGetBasicBlocks(Fn, BasicBlocks)

   Obtain all of the basic blocks in a function.

   This operates on a function value. The BasicBlocks parameter is a
   pointer to a pre-allocated array of LLVMBasicBlockRef of at least
   LLVMCountBasicBlocks() in length. This array is populated with
   LLVMBasicBlockRef instances.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BasicBlocks (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetBasicBlocks(LLVMValueRef Fn, LLVMBasicBlockRef * BasicBlocks)


.. py:function:: LLVMGetFirstBasicBlock(Fn)

   Obtain the first basic block in a function.

   The returned basic block can be used as an iterator. You will likely
   eventually call into LLVMGetNextBasicBlock() with it.

   See:
       llvm::Function::begin()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn)


.. py:function:: LLVMGetLastBasicBlock(Fn)

   Obtain the last basic block in a function.

   See:
       llvm::Function::end()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn)


.. py:function:: LLVMGetNextBasicBlock(BB)

   Advance a basic block iterator.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB)


.. py:function:: LLVMGetPreviousBasicBlock(BB)

   Go backwards in a basic block iterator.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB)


.. py:function:: LLVMGetEntryBasicBlock(Fn)

   Obtain the basic block that corresponds to the entry point of a
   function.

   See:
       llvm::Function::getEntryBlock()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn)


.. py:function:: LLVMInsertExistingBasicBlockAfterInsertBlock(Builder, BB)

   Insert the given basic block after the insertion point of the given builder.

   The insertion point must be valid.

   See:
       llvm::Function::BasicBlockListType::insertAfter()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInsertExistingBasicBlockAfterInsertBlock(LLVMBuilderRef Builder, LLVMBasicBlockRef BB)


.. py:function:: LLVMAppendExistingBasicBlock(Fn, BB)

   Append the given basic block to the basic block list of the given function.

   See:
       llvm::Function::BasicBlockListType::push_back()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAppendExistingBasicBlock(LLVMValueRef Fn, LLVMBasicBlockRef BB)


.. py:function:: LLVMCreateBasicBlockInContext(C, Name)

   Create a new basic block without inserting it into a function.

   See:
       llvm::BasicBlock::Create()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMCreateBasicBlockInContext(LLVMContextRef C, const char * Name)


.. py:function:: LLVMAppendBasicBlockInContext(C, Fn, Name)

   Append a basic block to the end of a function.

   See:
       llvm::BasicBlock::Create()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMAppendBasicBlockInContext(LLVMContextRef C, LLVMValueRef Fn, const char * Name)


.. py:function:: LLVMAppendBasicBlock(Fn, Name)

   Append a basic block to the end of a function using the global
   context.

   See:
       llvm::BasicBlock::Create()

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef Fn, const char * Name)


.. py:function:: LLVMInsertBasicBlockInContext(C, BB, Name)

   Insert a basic block in a function before another basic block.

   The function to add to is determined by the function of the
   passed basic block.

   See:
       llvm::BasicBlock::Create()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMInsertBasicBlockInContext(LLVMContextRef C, LLVMBasicBlockRef BB, const char * Name)


.. py:function:: LLVMInsertBasicBlock(InsertBeforeBB, Name)

   Insert a basic block in a function using the global context.

   See:
       llvm::BasicBlock::Create()

   Args:
       InsertBeforeBB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBB, const char * Name)


.. py:function:: LLVMDeleteBasicBlock(BB)

   Remove a basic block from a function and delete it.

   This deletes the basic block from its containing function and deletes
   the basic block itself.

   See:
       llvm::BasicBlock::eraseFromParent()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDeleteBasicBlock(LLVMBasicBlockRef BB)


.. py:function:: LLVMRemoveBasicBlockFromParent(BB)

   Remove a basic block from a function.

   This deletes the basic block from its containing function but keep
   the basic block alive.

   See:
       llvm::BasicBlock::removeFromParent()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveBasicBlockFromParent(LLVMBasicBlockRef BB)


.. py:function:: LLVMMoveBasicBlockBefore(BB, MovePos)

   Move a basic block to before another one.

   See:
       llvm::BasicBlock::moveBefore()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MovePos (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveBasicBlockBefore(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos)


.. py:function:: LLVMMoveBasicBlockAfter(BB, MovePos)

   Move a basic block to after another one.

   See:
       llvm::BasicBlock::moveAfter()

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MovePos (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveBasicBlockAfter(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos)


.. py:function:: LLVMGetFirstInstruction(BB)

   Obtain the first instruction in a basic block.

   The returned LLVMValueRef corresponds to a llvm::Instruction
   instance.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB)


.. py:function:: LLVMGetLastInstruction(BB)

   Obtain the last instruction in a basic block.

   The returned LLVMValueRef corresponds to an LLVM:Instruction.

   Args:
       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB)


.. py:function:: LLVMHasMetadata(Val)

   Determine whether an instruction has any metadata attached.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMHasMetadata(LLVMValueRef Val)


.. py:function:: LLVMGetMetadata(Val, KindID)

   Return metadata associated with an instruction value.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetMetadata(LLVMValueRef Val, unsigned int KindID)


.. py:function:: LLVMSetMetadata(Val, KindID, Node)

   Set metadata associated with an instruction value.

   Args:
       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

       Node (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetMetadata(LLVMValueRef Val, unsigned int KindID, LLVMValueRef Node)


.. py:function:: LLVMInstructionGetAllMetadataOtherThanDebugLoc(Instr, NumEntries)

   Returns the metadata associated with an instruction value, but filters out
   all the debug locations.

   See:
       llvm::Instruction::getAllMetadataOtherThanDebugLoc()

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumEntries (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueMetadataEntry * LLVMInstructionGetAllMetadataOtherThanDebugLoc(LLVMValueRef Instr, size_t * NumEntries)


.. py:function:: LLVMGetInstructionParent(Inst)

   Obtain the basic block to which an instruction belongs.

   See:
       llvm::Instruction::getParent()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst)


.. py:function:: LLVMGetNextInstruction(Inst)

   Obtain the instruction that occurs after the one specified.

   The next instruction will be from the same basic block.

   If this is the last instruction in a basic block, NULL will be
   returned.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst)


.. py:function:: LLVMGetPreviousInstruction(Inst)

   Obtain the instruction that occurred before this one.

   If the instruction is the first instruction in a basic block, NULL
   will be returned.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst)


.. py:function:: LLVMInstructionRemoveFromParent(Inst)

   Remove an instruction.

   The instruction specified is removed from its containing building
   block but is kept alive.

   See:
       llvm::Instruction::removeFromParent()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInstructionRemoveFromParent(LLVMValueRef Inst)


.. py:function:: LLVMInstructionEraseFromParent(Inst)

   Remove and delete an instruction.

   The instruction specified is removed from its containing building
   block and then deleted.

   See:
       llvm::Instruction::eraseFromParent()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInstructionEraseFromParent(LLVMValueRef Inst)


.. py:function:: LLVMDeleteInstruction(Inst)

   Delete an instruction.

   The instruction specified is deleted. It must have previously been
   removed from its containing building block.

   See:
       llvm::Value::deleteValue()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDeleteInstruction(LLVMValueRef Inst)


.. py:function:: LLVMGetInstructionOpcode(Inst)

   Obtain the code opcode for an individual instruction.

   See:
       llvm::Instruction::getOpCode()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMOpcode`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOpcode LLVMGetInstructionOpcode(LLVMValueRef Inst)


.. py:function:: LLVMGetICmpPredicate(Inst)

   Obtain the predicate of an instruction.

   This is only valid for instructions that correspond to llvm::ICmpInst.

   See:
       llvm::ICmpInst::getPredicate()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMIntPredicate`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef Inst)


.. py:function:: LLVMGetICmpSameSign(Inst)

   Get whether or not an icmp instruction has the samesign flag.

   This is only valid for instructions that correspond to llvm::ICmpInst.

   See:
       llvm::ICmpInst::hasSameSign()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetICmpSameSign(LLVMValueRef Inst)


.. py:function:: LLVMSetICmpSameSign(Inst, SameSign)

   Set the samesign flag on an icmp instruction.

   This is only valid for instructions that correspond to llvm::ICmpInst.

   See:
       llvm::ICmpInst::setSameSign()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SameSign (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetICmpSameSign(LLVMValueRef Inst, LLVMBool SameSign)


.. py:function:: LLVMGetFCmpPredicate(Inst)

   Obtain the float predicate of an instruction.

   This is only valid for instructions that correspond to llvm::FCmpInst.

   See:
       llvm::FCmpInst::getPredicate()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMRealPredicate`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef Inst)


.. py:function:: LLVMInstructionClone(Inst)

   Create a copy of 'this' instruction that is identical in all ways except the following:   * The instruction has no parent   * The instruction has no name

   Create a copy of 'this' instruction that is identical in all ways
   except the following:
     * The instruction has no parent
     * The instruction has no name

   See:
       llvm::Instruction::clone()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMInstructionClone(LLVMValueRef Inst)


.. py:function:: LLVMIsATerminatorInst(Inst)

   Determine whether an instruction is a terminator.

   This routine is named to
   be compatible with historical functions that did this by querying the
   underlying C++ type.

   See:
       llvm::Instruction::isTerminator()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMIsATerminatorInst(LLVMValueRef Inst)


.. py:function:: LLVMGetFirstDbgRecord(Inst)

   Obtain the first debug record attached to an instruction.

   Use LLVMGetNextDbgRecord() and LLVMGetPreviousDbgRecord() to traverse the
   sequence of DbgRecords.

   Return the first DbgRecord attached to Inst or NULL if there are none.

   See:
       llvm::Instruction::getDbgRecordRange()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMGetFirstDbgRecord(LLVMValueRef Inst)


.. py:function:: LLVMGetLastDbgRecord(Inst)

   Obtain the last debug record attached to an instruction.

   Return the last DbgRecord attached to Inst or NULL if there are none.

   See:
       llvm::Instruction::getDbgRecordRange()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMGetLastDbgRecord(LLVMValueRef Inst)


.. py:function:: LLVMGetNextDbgRecord(DbgRecord)

   Obtain the next DbgRecord in the sequence or NULL if there are no more.

   See:
       llvm::Instruction::getDbgRecordRange()

   Args:
       DbgRecord (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMGetNextDbgRecord(LLVMDbgRecordRef DbgRecord)


.. py:function:: LLVMGetPreviousDbgRecord(DbgRecord)

   Obtain the previous DbgRecord in the sequence or NULL if there are no more.

   See:
       llvm::Instruction::getDbgRecordRange()

   Args:
       DbgRecord (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMGetPreviousDbgRecord(LLVMDbgRecordRef DbgRecord)


.. py:function:: LLVMDbgRecordGetDebugLoc(Rec)

   Get the debug location attached to the debug record.

   See:
       llvm::DbgRecord::getDebugLoc()

   Args:
       Rec (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDbgRecordGetDebugLoc(LLVMDbgRecordRef Rec)


.. py:function:: LLVMDbgRecordGetKind(Rec)

   Get the debug location attached to the debug record.

   See:
       llvm::DbgRecord::getDebugLoc()

   Args:
       Rec (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMDbgRecordKind`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordKind LLVMDbgRecordGetKind(LLVMDbgRecordRef Rec)


.. py:function:: LLVMDbgVariableRecordGetValue(Rec, OpIdx)

   Get the value of the DbgVariableRecord.

   See:
       llvm::DbgVariableRecord::getValue()

   Args:
       Rec (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OpIdx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMDbgVariableRecordGetValue(LLVMDbgRecordRef Rec, unsigned int OpIdx)


.. py:function:: LLVMDbgVariableRecordGetVariable(Rec)

   Get the debug info variable of the DbgVariableRecord.

   See:
       llvm::DbgVariableRecord::getVariable()

   Args:
       Rec (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDbgVariableRecordGetVariable(LLVMDbgRecordRef Rec)


.. py:function:: LLVMDbgVariableRecordGetExpression(Rec)

   Get the debug info expression of the DbgVariableRecord.

   See:
       llvm::DbgVariableRecord::getExpression()

   Args:
       Rec (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDbgVariableRecordGetExpression(LLVMDbgRecordRef Rec)


.. py:function:: LLVMGetNumArgOperands(Instr)

   Obtain the argument count for a call instruction.

   This expects an LLVMValueRef that corresponds to a llvm::CallInst,
   llvm::InvokeInst, or llvm:FuncletPadInst.

   See:
       llvm::CallInst::getNumArgOperands()

   See:
       llvm::InvokeInst::getNumArgOperands()

   See:
       llvm::FuncletPadInst::getNumArgOperands()

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumArgOperands(LLVMValueRef Instr)


.. py:function:: LLVMSetInstructionCallConv(Instr, CC)

   Set the calling convention for a call instruction.

   This expects an LLVMValueRef that corresponds to a llvm::CallInst or
   llvm::InvokeInst.

   See:
       llvm::CallInst::setCallingConv()

   See:
       llvm::InvokeInst::setCallingConv()

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CC (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned int CC)


.. py:function:: LLVMGetInstructionCallConv(Instr)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetInstructionCallConv(LLVMValueRef Instr)


.. py:function:: LLVMSetInstrParamAlignment(Instr, Idx, Align)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       Align (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetInstrParamAlignment(LLVMValueRef Instr, LLVMAttributeIndex Idx, unsigned int Align)


.. py:function:: LLVMAddCallSiteAttribute(C, Idx, A)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddCallSiteAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, LLVMAttributeRef A)


.. py:function:: LLVMGetCallSiteAttributeCount(C, Idx)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetCallSiteAttributeCount(LLVMValueRef C, LLVMAttributeIndex Idx)


.. py:function:: LLVMGetCallSiteAttributes(C, Idx, Attrs)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       Attrs (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetCallSiteAttributes(LLVMValueRef C, LLVMAttributeIndex Idx, LLVMAttributeRef * Attrs)


.. py:function:: LLVMGetCallSiteEnumAttribute(C, Idx, KindID)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMGetCallSiteEnumAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, unsigned int KindID)


.. py:function:: LLVMGetCallSiteStringAttribute(C, Idx, K, KLen)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       K (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KLen (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAttributeRef LLVMGetCallSiteStringAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, const char * K, unsigned int KLen)


.. py:function:: LLVMRemoveCallSiteEnumAttribute(C, Idx, KindID)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       KindID (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveCallSiteEnumAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, unsigned int KindID)


.. py:function:: LLVMRemoveCallSiteStringAttribute(C, Idx, K, KLen)

   Obtain the calling convention for a call instruction.

   This is the opposite of LLVMSetInstructionCallConv(). Reads its
   usage.

   See:
       :py:obj:`~.LLVMSetInstructionCallConv`

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       K (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       KLen (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemoveCallSiteStringAttribute(LLVMValueRef C, LLVMAttributeIndex Idx, const char * K, unsigned int KLen)


.. py:function:: LLVMGetCalledFunctionType(C)

   Obtain the function type called by this instruction.

   See:
       llvm::CallBase::getFunctionType()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetCalledFunctionType(LLVMValueRef C)


.. py:function:: LLVMGetCalledValue(Instr)

   Obtain the pointer to the function invoked by this instruction.

   This expects an LLVMValueRef that corresponds to a llvm::CallInst or
   llvm::InvokeInst.

   See:
       llvm::CallInst::getCalledOperand()

   See:
       llvm::InvokeInst::getCalledOperand()

   Args:
       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetCalledValue(LLVMValueRef Instr)


.. py:function:: LLVMGetNumOperandBundles(C)

   Obtain the number of operand bundles attached to this instruction.

   This only works on llvm::CallInst and llvm::InvokeInst instructions.

   See:
       llvm::CallBase::getNumOperandBundles()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumOperandBundles(LLVMValueRef C)


.. py:function:: LLVMGetOperandBundleAtIndex(C, Index)

   Obtain the operand bundle attached to this instruction at the given index.

   Use LLVMDisposeOperandBundle to free the operand bundle.

   This only works on llvm::CallInst and llvm::InvokeInst instructions.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOperandBundleRef LLVMGetOperandBundleAtIndex(LLVMValueRef C, unsigned int Index)


.. py:function:: LLVMIsTailCall(CallInst)

   Obtain whether a call instruction is a tail call.

   This only works on llvm::CallInst instructions.

   See:
       llvm::CallInst::isTailCall()

   Args:
       CallInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsTailCall(LLVMValueRef CallInst)


.. py:function:: LLVMSetTailCall(CallInst, IsTailCall)

   Set whether a call instruction is a tail call.

   This only works on llvm::CallInst instructions.

   See:
       llvm::CallInst::setTailCall()

   Args:
       CallInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsTailCall (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTailCall(LLVMValueRef CallInst, LLVMBool IsTailCall)


.. py:function:: LLVMGetTailCallKind(CallInst)

   Obtain a tail call kind of the call instruction.

   See:
       llvm::CallInst::setTailCallKind()

   Args:
       CallInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMTailCallKind`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTailCallKind LLVMGetTailCallKind(LLVMValueRef CallInst)


.. py:function:: LLVMSetTailCallKind(CallInst, kind)

   Set the call kind of the call instruction.

   See:
       llvm::CallInst::getTailCallKind()

   Args:
       CallInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       kind (:py:obj:`~.LLVMTailCallKind`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTailCallKind(LLVMValueRef CallInst, LLVMTailCallKind kind)


.. py:function:: LLVMGetNormalDest(InvokeInst)

   Return the normal destination basic block.

   This only works on llvm::InvokeInst instructions.

   See:
       llvm::InvokeInst::getNormalDest()

   Args:
       InvokeInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetNormalDest(LLVMValueRef InvokeInst)


.. py:function:: LLVMGetUnwindDest(InvokeInst)

   Return the unwind destination basic block.

   Works on llvm::InvokeInst, llvm::CleanupReturnInst, and
   llvm::CatchSwitchInst instructions.

   See:
       llvm::InvokeInst::getUnwindDest()

   See:
       llvm::CleanupReturnInst::getUnwindDest()

   See:
       llvm::CatchSwitchInst::getUnwindDest()

   Args:
       InvokeInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetUnwindDest(LLVMValueRef InvokeInst)


.. py:function:: LLVMSetNormalDest(InvokeInst, B)

   Set the normal destination basic block.

   This only works on llvm::InvokeInst instructions.

   See:
       llvm::InvokeInst::setNormalDest()

   Args:
       InvokeInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetNormalDest(LLVMValueRef InvokeInst, LLVMBasicBlockRef B)


.. py:function:: LLVMSetUnwindDest(InvokeInst, B)

   Set the unwind destination basic block.

   Works on llvm::InvokeInst, llvm::CleanupReturnInst, and
   llvm::CatchSwitchInst instructions.

   See:
       llvm::InvokeInst::setUnwindDest()

   See:
       llvm::CleanupReturnInst::setUnwindDest()

   See:
       llvm::CatchSwitchInst::setUnwindDest()

   Args:
       InvokeInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetUnwindDest(LLVMValueRef InvokeInst, LLVMBasicBlockRef B)


.. py:function:: LLVMGetCallBrDefaultDest(CallBr)

   Get the default destination of a CallBr instruction.

   See:
       llvm::CallBrInst::getDefaultDest()

   Args:
       CallBr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetCallBrDefaultDest(LLVMValueRef CallBr)


.. py:function:: LLVMGetCallBrNumIndirectDests(CallBr)

   Get the number of indirect destinations of a CallBr instruction.

   See:
       llvm::CallBrInst::getNumIndirectDests()

   Args:
       CallBr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetCallBrNumIndirectDests(LLVMValueRef CallBr)


.. py:function:: LLVMGetCallBrIndirectDest(CallBr, Idx)

   Get the indirect destination of a CallBr instruction at the given index.

   See:
       llvm::CallBrInst::getIndirectDest()

   Args:
       CallBr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetCallBrIndirectDest(LLVMValueRef CallBr, unsigned int Idx)


.. py:function:: LLVMGetNumSuccessors(Term)

   Return the number of successors that this terminator has.

   See:
       llvm::Instruction::getNumSuccessors

   Args:
       Term (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumSuccessors(LLVMValueRef Term)


.. py:function:: LLVMGetSuccessor(Term, i)

   Return the specified successor.

   See:
       llvm::Instruction::getSuccessor

   Args:
       Term (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef Term, unsigned int i)


.. py:function:: LLVMSetSuccessor(Term, i, block)

   Update the specified successor to point at the provided block.

   See:
       llvm::Instruction::setSuccessor

   Args:
       Term (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

       block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetSuccessor(LLVMValueRef Term, unsigned int i, LLVMBasicBlockRef block)


.. py:function:: LLVMIsConditional(Branch)

   Return if an instruction is a conditional branch.

   Deprecated: Use LLVMIsACondBrInst instead.

   Args:
       Branch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsConditional(LLVMValueRef Branch)


.. py:function:: LLVMGetCondition(Branch)

   Return the condition of a branch instruction.

   This only works on llvm::CondBrInst instructions.

   See:
       llvm::CondBrInst::getCondition

   Args:
       Branch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetCondition(LLVMValueRef Branch)


.. py:function:: LLVMSetCondition(Branch, Cond)

   Set the condition of a branch instruction.

   This only works on llvm::CondBrInst instructions.

   See:
       llvm::CondBrInst::setCondition

   Args:
       Branch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Cond (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCondition(LLVMValueRef Branch, LLVMValueRef Cond)


.. py:function:: LLVMGetSwitchDefaultDest(SwitchInstr)

   Obtain the default destination basic block of a switch instruction.

   This only works on llvm::SwitchInst instructions.

   See:
       llvm::SwitchInst::getDefaultDest()

   Args:
       SwitchInstr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetSwitchDefaultDest(LLVMValueRef SwitchInstr)


.. py:function:: LLVMGetSwitchCaseValue(SwitchInstr, i)

   Obtain the case value for a successor of a switch instruction.

   i corresponds
   to the successor index. The first successor is the default destination, so i
   must be greater than zero.

   This only works on llvm::SwitchInst instructions.

   See:
       llvm::SwitchInst::CaseHandle::getCaseValue()

   Args:
       SwitchInstr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetSwitchCaseValue(LLVMValueRef SwitchInstr, unsigned int i)


.. py:function:: LLVMSetSwitchCaseValue(SwitchInstr, i, CaseValue)

   Set the case value for a successor of a switch instruction.

   i corresponds to
   the successor index. The first successor is the default destination, so i
   must be greater than zero.

   This only works on llvm::SwitchInst instructions.

   See:
       llvm::SwitchInst::CaseHandle::setValue()

   Args:
       SwitchInstr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

       CaseValue (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetSwitchCaseValue(LLVMValueRef SwitchInstr, unsigned int i, LLVMValueRef CaseValue)


.. py:function:: LLVMGetAllocatedType(Alloca)

   Obtain the type that is being allocated by the alloca instruction.

   Args:
       Alloca (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetAllocatedType(LLVMValueRef Alloca)


.. py:function:: LLVMIsInBounds(GEP)

   Check whether the given GEP operator is inbounds.

   Args:
       GEP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsInBounds(LLVMValueRef GEP)


.. py:function:: LLVMSetIsInBounds(GEP, InBounds)

   Set the given GEP instruction to be inbounds or not.

   Args:
       GEP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       InBounds (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetIsInBounds(LLVMValueRef GEP, LLVMBool InBounds)


.. py:function:: LLVMGetGEPSourceElementType(GEP)

   Get the source element type of the given GEP operator.

   Args:
       GEP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMGetGEPSourceElementType(LLVMValueRef GEP)


.. py:function:: LLVMGEPGetNoWrapFlags(GEP)

   Get the no-wrap related flags for the given GEP instruction.

   See:
       llvm::GetElementPtrInst::getNoWrapFlags

   Args:
       GEP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMGEPNoWrapFlags LLVMGEPGetNoWrapFlags(LLVMValueRef GEP)


.. py:function:: LLVMGEPSetNoWrapFlags(GEP, NoWrapFlags)

   Set the no-wrap related flags for the given GEP instruction.

   See:
       llvm::GetElementPtrInst::setNoWrapFlags

   Args:
       GEP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NoWrapFlags (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGEPSetNoWrapFlags(LLVMValueRef GEP, LLVMGEPNoWrapFlags NoWrapFlags)


.. py:function:: LLVMAddIncoming(PhiNode, IncomingValues, IncomingBlocks, Count)

   Add an incoming value to the end of a PHI list.

   Args:
       PhiNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IncomingValues (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IncomingBlocks (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Count (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef * IncomingValues, LLVMBasicBlockRef * IncomingBlocks, unsigned int Count)


.. py:function:: LLVMCountIncoming(PhiNode)

   Obtain the number of incoming basic blocks to a PHI node.

   Args:
       PhiNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCountIncoming(LLVMValueRef PhiNode)


.. py:function:: LLVMGetIncomingValue(PhiNode, Index)

   Obtain an incoming value to a PHI node as an LLVMValueRef.

   Args:
       PhiNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, unsigned int Index)


.. py:function:: LLVMGetIncomingBlock(PhiNode, Index)

   Obtain an incoming value to a PHI node as an LLVMBasicBlockRef.

   Args:
       PhiNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, unsigned int Index)


.. py:function:: LLVMGetNumIndices(Inst)

   Obtain the number of indices.

   NB: This also works on GEP operators.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumIndices(LLVMValueRef Inst)


.. py:function:: LLVMGetIndices(Inst)

   Obtain the indices as an array.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const unsigned int * LLVMGetIndices(LLVMValueRef Inst)


.. py:function:: LLVMCreateBuilderInContext(C)

   An instruction builder represents a point within a basic block and is
   the exclusive means of building instructions using the C interface.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBuilderRef LLVMCreateBuilderInContext(LLVMContextRef C)


.. py:function:: LLVMCreateBuilder()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBuilderRef LLVMCreateBuilder()


.. py:function:: LLVMPositionBuilder(Builder, Block, Instr)

   Set the builder position before Instr but after any attached debug records,
   or if Instr is null set the position to the end of Block.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block, LLVMValueRef Instr)


.. py:function:: LLVMPositionBuilderBeforeDbgRecords(Builder, Block, Inst)

   Set the builder position before Instr and any attached debug records,
   or if Instr is null set the position to the end of Block.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPositionBuilderBeforeDbgRecords(LLVMBuilderRef Builder, LLVMBasicBlockRef Block, LLVMValueRef Inst)


.. py:function:: LLVMPositionBuilderBefore(Builder, Instr)

   Set the builder position before Instr but after any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr)


.. py:function:: LLVMPositionBuilderBeforeInstrAndDbgRecords(Builder, Instr)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPositionBuilderBeforeInstrAndDbgRecords(LLVMBuilderRef Builder, LLVMValueRef Instr)


.. py:function:: LLVMPositionBuilderAtEnd(Builder, Block)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block)


.. py:function:: LLVMGetInsertBlock(Builder)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder)


.. py:function:: LLVMClearInsertionPosition(Builder)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMClearInsertionPosition(LLVMBuilderRef Builder)


.. py:function:: LLVMInsertIntoBuilder(Builder, Instr)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInsertIntoBuilder(LLVMBuilderRef Builder, LLVMValueRef Instr)


.. py:function:: LLVMInsertIntoBuilderWithName(Builder, Instr, Name)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInsertIntoBuilderWithName(LLVMBuilderRef Builder, LLVMValueRef Instr, const char * Name)


.. py:function:: LLVMDisposeBuilder(Builder)

   Set the builder position before Instr and any attached debug records.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeBuilder(LLVMBuilderRef Builder)


.. py:function:: LLVMGetCurrentDebugLocation2(Builder)

   Get location information used by debugging information.

   See:
       llvm::IRBuilder::getCurrentDebugLocation()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMGetCurrentDebugLocation2(LLVMBuilderRef Builder)


.. py:function:: LLVMSetCurrentDebugLocation2(Builder, Loc)

   Set location information used by debugging information.

   To clear the location metadata of the given instruction, pass NULL to ``Loc.``

   See:
       llvm::IRBuilder::SetCurrentDebugLocation()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Loc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCurrentDebugLocation2(LLVMBuilderRef Builder, LLVMMetadataRef Loc)


.. py:function:: LLVMSetInstDebugLocation(Builder, Inst)

   Attempts to set the debug location for the given instruction using the
   current debug location for the given builder.

   If the builder has no current
   debug location, this function is a no-op.

   Deprecated:
       LLVMSetInstDebugLocation is deprecated in favor of the more general
       LLVMAddMetadataToInst.

   See:
       llvm::IRBuilder::SetInstDebugLocation()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetInstDebugLocation(LLVMBuilderRef Builder, LLVMValueRef Inst)


.. py:function:: LLVMAddMetadataToInst(Builder, Inst)

   Adds the metadata registered with the given builder to the given instruction.

   See:
       llvm::IRBuilder::AddMetadataToInst()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddMetadataToInst(LLVMBuilderRef Builder, LLVMValueRef Inst)


.. py:function:: LLVMBuilderGetDefaultFPMathTag(Builder)

   Get the dafult floating-point math metadata for a given builder.

   See:
       llvm::IRBuilder::getDefaultFPMathTag()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMBuilderGetDefaultFPMathTag(LLVMBuilderRef Builder)


.. py:function:: LLVMBuilderSetDefaultFPMathTag(Builder, FPMathTag)

   Set the default floating-point math metadata for the given builder.

   To clear the metadata, pass NULL to ``FPMathTag.``

   See:
       llvm::IRBuilder::setDefaultFPMathTag()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       FPMathTag (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMBuilderSetDefaultFPMathTag(LLVMBuilderRef Builder, LLVMMetadataRef FPMathTag)


.. py:function:: LLVMGetBuilderContext(Builder)

   Obtain the context to which this builder is associated.

   See:
       llvm::IRBuilder::getContext()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMContextRef LLVMGetBuilderContext(LLVMBuilderRef Builder)


.. py:function:: LLVMSetCurrentDebugLocation(Builder, L)

   Deprecated: Passing the NULL location will crash.

   Use LLVMGetCurrentDebugLocation2 instead.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       L (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCurrentDebugLocation(LLVMBuilderRef Builder, LLVMValueRef L)


.. py:function:: LLVMGetCurrentDebugLocation(Builder)

   Deprecated: Returning the NULL location will crash.

   Use LLVMGetCurrentDebugLocation2 instead.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetCurrentDebugLocation(LLVMBuilderRef Builder)


.. py:function:: LLVMBuildRetVoid(arg0)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef)


.. py:function:: LLVMBuildRet(arg0, V)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildRet(LLVMBuilderRef, LLVMValueRef V)


.. py:function:: LLVMBuildAggregateRet(arg0, RetVals, N)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RetVals (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       N (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAggregateRet(LLVMBuilderRef, LLVMValueRef * RetVals, unsigned int N)


.. py:function:: LLVMBuildBr(arg0, Dest)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildBr(LLVMBuilderRef, LLVMBasicBlockRef Dest)


.. py:function:: LLVMBuildCondBr(arg0, If, Then, Else)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       If (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Then (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Else (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef, LLVMValueRef If, LLVMBasicBlockRef Then, LLVMBasicBlockRef Else)


.. py:function:: LLVMBuildSwitch(arg0, V, Else, NumCases)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Else (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumCases (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef, LLVMValueRef V, LLVMBasicBlockRef Else, unsigned int NumCases)


.. py:function:: LLVMBuildIndirectBr(B, Addr, NumDests)

   Terminators

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Addr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumDests (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIndirectBr(LLVMBuilderRef B, LLVMValueRef Addr, unsigned int NumDests)


.. py:function:: LLVMBuildCallBr(B, Ty, Fn, DefaultDest, IndirectDests, NumIndirectDests, Args, NumArgs, Bundles, NumBundles, Name)

   Terminators

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DefaultDest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IndirectDests (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndirectDests (:py:obj:`~.int`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Bundles (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumBundles (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCallBr(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMBasicBlockRef DefaultDest, LLVMBasicBlockRef * IndirectDests, unsigned int NumIndirectDests, LLVMValueRef * Args, unsigned int NumArgs, LLVMOperandBundleRef * Bundles, unsigned int NumBundles, const char * Name)


.. py:function:: LLVMBuildInvoke2(arg0, Ty, Fn, Args, NumArgs, Then, Catch, Name)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Then (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Catch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildInvoke2(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMValueRef * Args, unsigned int NumArgs, LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch, const char * Name)


.. py:function:: LLVMBuildInvokeWithOperandBundles(arg0, Ty, Fn, Args, NumArgs, Then, Catch, Bundles, NumBundles, Name)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Then (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Catch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Bundles (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumBundles (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildInvokeWithOperandBundles(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Fn, LLVMValueRef * Args, unsigned int NumArgs, LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch, LLVMOperandBundleRef * Bundles, unsigned int NumBundles, const char * Name)


.. py:function:: LLVMBuildUnreachable(arg0)

   Terminators

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef)


.. py:function:: LLVMBuildResume(B, Exn)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Exn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildResume(LLVMBuilderRef B, LLVMValueRef Exn)


.. py:function:: LLVMBuildLandingPad(B, Ty, PersFn, NumClauses, Name)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       PersFn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumClauses (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildLandingPad(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef PersFn, unsigned int NumClauses, const char * Name)


.. py:function:: LLVMBuildCleanupRet(B, CatchPad, BB)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CatchPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCleanupRet(LLVMBuilderRef B, LLVMValueRef CatchPad, LLVMBasicBlockRef BB)


.. py:function:: LLVMBuildCatchRet(B, CatchPad, BB)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CatchPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCatchRet(LLVMBuilderRef B, LLVMValueRef CatchPad, LLVMBasicBlockRef BB)


.. py:function:: LLVMBuildCatchPad(B, ParentPad, Args, NumArgs, Name)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ParentPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCatchPad(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMValueRef * Args, unsigned int NumArgs, const char * Name)


.. py:function:: LLVMBuildCleanupPad(B, ParentPad, Args, NumArgs, Name)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ParentPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCleanupPad(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMValueRef * Args, unsigned int NumArgs, const char * Name)


.. py:function:: LLVMBuildCatchSwitch(B, ParentPad, UnwindBB, NumHandlers, Name)

   Exception Handling

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ParentPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       UnwindBB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumHandlers (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCatchSwitch(LLVMBuilderRef B, LLVMValueRef ParentPad, LLVMBasicBlockRef UnwindBB, unsigned int NumHandlers, const char * Name)


.. py:function:: LLVMAddCase(Switch, OnVal, Dest)

   Add a case to the switch instruction

   Args:
       Switch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OnVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal, LLVMBasicBlockRef Dest)


.. py:function:: LLVMAddDestination(IndirectBr, Dest)

   Add a destination to the indirectbr instruction

   Args:
       IndirectBr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddDestination(LLVMValueRef IndirectBr, LLVMBasicBlockRef Dest)


.. py:function:: LLVMGetNumClauses(LandingPad)

   Get the number of clauses on the landingpad instruction

   Args:
       LandingPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumClauses(LLVMValueRef LandingPad)


.. py:function:: LLVMGetClause(LandingPad, Idx)

   Get the value of the clause at index Idx on the landingpad instruction

   Args:
       LandingPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetClause(LLVMValueRef LandingPad, unsigned int Idx)


.. py:function:: LLVMAddClause(LandingPad, ClauseVal)

   Add a catch or filter clause to the landingpad instruction

   Args:
       LandingPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ClauseVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddClause(LLVMValueRef LandingPad, LLVMValueRef ClauseVal)


.. py:function:: LLVMIsCleanup(LandingPad)

   Get the 'cleanup' flag in the landingpad instruction

   Args:
       LandingPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsCleanup(LLVMValueRef LandingPad)


.. py:function:: LLVMSetCleanup(LandingPad, Val)

   Set the 'cleanup' flag in the landingpad instruction

   Args:
       LandingPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCleanup(LLVMValueRef LandingPad, LLVMBool Val)


.. py:function:: LLVMAddHandler(CatchSwitch, Dest)

   Add a destination to the catchswitch instruction

   Args:
       CatchSwitch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddHandler(LLVMValueRef CatchSwitch, LLVMBasicBlockRef Dest)


.. py:function:: LLVMGetNumHandlers(CatchSwitch)

   Get the number of handlers on the catchswitch instruction

   Args:
       CatchSwitch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumHandlers(LLVMValueRef CatchSwitch)


.. py:function:: LLVMGetHandlers(CatchSwitch, Handlers)

   Obtain the basic blocks acting as handlers for a catchswitch instruction.

   The Handlers parameter should point to a pre-allocated array of
   LLVMBasicBlockRefs at least LLVMGetNumHandlers() large. On return, the
   first LLVMGetNumHandlers() entries in the array will be populated
   with LLVMBasicBlockRef instances.

   Args:
       CatchSwitch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The catchswitch instruction to operate on.

       Handlers (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Memory address of an array to be filled with basic blocks.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMGetHandlers(LLVMValueRef CatchSwitch, LLVMBasicBlockRef * Handlers)


.. py:function:: LLVMGetArgOperand(Funclet, i)

   Get the number of funcletpad arguments.

   Args:
       Funclet (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetArgOperand(LLVMValueRef Funclet, unsigned int i)


.. py:function:: LLVMSetArgOperand(Funclet, i, value)

   Set a funcletpad argument at the given index.

   Args:
       Funclet (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       i (:py:obj:`~.int`):
           (undocumented)

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetArgOperand(LLVMValueRef Funclet, unsigned int i, LLVMValueRef value)


.. py:function:: LLVMGetParentCatchSwitch(CatchPad)

   Get the parent catchswitch instruction of a catchpad instruction.

   This only works on llvm::CatchPadInst instructions.

   See:
       llvm::CatchPadInst::getCatchSwitch()

   Args:
       CatchPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMGetParentCatchSwitch(LLVMValueRef CatchPad)


.. py:function:: LLVMSetParentCatchSwitch(CatchPad, CatchSwitch)

   Set the parent catchswitch instruction of a catchpad instruction.

   This only works on llvm::CatchPadInst instructions.

   See:
       llvm::CatchPadInst::setCatchSwitch()

   Args:
       CatchPad (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CatchSwitch (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetParentCatchSwitch(LLVMValueRef CatchPad, LLVMValueRef CatchSwitch)


.. py:function:: LLVMBuildAdd(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNSWAdd(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNSWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNUWAdd(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNUWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFAdd(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildSub(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNSWSub(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNSWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNUWSub(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNUWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFSub(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildMul(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNSWMul(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNSWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNUWMul(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNUWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFMul(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildUDiv(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildExactUDiv(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildExactUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildSDiv(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildExactSDiv(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildExactSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFDiv(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildURem(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildURem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildSRem(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFRem(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildShl(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildShl(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildLShr(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildLShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildAShr(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildAnd(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAnd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildOr(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildOr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildXor(arg0, LHS, RHS, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildXor(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildBinOp(B, Op, LHS, RHS, Name)

   Arithmetic

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Op (:py:obj:`~.LLVMOpcode`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildBinOp(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildNeg(arg0, V, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNeg(LLVMBuilderRef, LLVMValueRef V, const char * Name)


.. py:function:: LLVMBuildNSWNeg(B, V, Name)

   Arithmetic

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNSWNeg(LLVMBuilderRef B, LLVMValueRef V, const char * Name)


.. py:function:: LLVMBuildNUWNeg(B, V, Name)

   (No short description, might be part of a group.)

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNUWNeg(LLVMBuilderRef B, LLVMValueRef V, const char * Name)


.. py:function:: LLVMBuildFNeg(arg0, V, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFNeg(LLVMBuilderRef, LLVMValueRef V, const char * Name)


.. py:function:: LLVMBuildNot(arg0, V, Name)

   Arithmetic

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildNot(LLVMBuilderRef, LLVMValueRef V, const char * Name)


.. py:function:: LLVMGetNUW(ArithInst)

   Arithmetic

   Args:
       ArithInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetNUW(LLVMValueRef ArithInst)


.. py:function:: LLVMSetNUW(ArithInst, HasNUW)

   Arithmetic

   Args:
       ArithInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       HasNUW (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetNUW(LLVMValueRef ArithInst, LLVMBool HasNUW)


.. py:function:: LLVMGetNSW(ArithInst)

   Arithmetic

   Args:
       ArithInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetNSW(LLVMValueRef ArithInst)


.. py:function:: LLVMSetNSW(ArithInst, HasNSW)

   Arithmetic

   Args:
       ArithInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       HasNSW (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetNSW(LLVMValueRef ArithInst, LLVMBool HasNSW)


.. py:function:: LLVMGetExact(DivOrShrInst)

   Arithmetic

   Args:
       DivOrShrInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetExact(LLVMValueRef DivOrShrInst)


.. py:function:: LLVMSetExact(DivOrShrInst, IsExact)

   Arithmetic

   Args:
       DivOrShrInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsExact (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetExact(LLVMValueRef DivOrShrInst, LLVMBool IsExact)


.. py:function:: LLVMGetNNeg(NonNegInst)

   Gets if the instruction has the non-negative flag set.

   Only valid for zext instructions.

   Args:
       NonNegInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetNNeg(LLVMValueRef NonNegInst)


.. py:function:: LLVMSetNNeg(NonNegInst, IsNonNeg)

   Sets the non-negative flag for the instruction.

   Only valid for zext instructions.

   Args:
       NonNegInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsNonNeg (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetNNeg(LLVMValueRef NonNegInst, LLVMBool IsNonNeg)


.. py:function:: LLVMGetFastMathFlags(FPMathInst)

   Get the flags for which fast-math-style optimizations are allowed for this
   value.

   Only valid on floating point instructions.

   See:
       :py:obj:`~.LLVMCanValueUseFastMathFlags`

   Args:
       FPMathInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMFastMathFlags LLVMGetFastMathFlags(LLVMValueRef FPMathInst)


.. py:function:: LLVMSetFastMathFlags(FPMathInst, FMF)

   Sets the flags for which fast-math-style optimizations are allowed for this
   value.

   Only valid on floating point instructions.

   See:
       :py:obj:`~.LLVMCanValueUseFastMathFlags`

   Args:
       FPMathInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       FMF (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetFastMathFlags(LLVMValueRef FPMathInst, LLVMFastMathFlags FMF)


.. py:function:: LLVMCanValueUseFastMathFlags(Inst)

   Check if a given value can potentially have fast math flags.

   Will return true for floating point arithmetic instructions, and for select,
   phi, and call instructions whose type is a floating point type, or a vector
   or array thereof. See https://llvm.org/docs/LangRef.html:py:obj:`~.fast`-math-flags

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCanValueUseFastMathFlags(LLVMValueRef Inst)


.. py:function:: LLVMGetIsDisjoint(Inst)

   Gets whether the instruction has the disjoint flag set.

   Only valid for or instructions.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetIsDisjoint(LLVMValueRef Inst)


.. py:function:: LLVMSetIsDisjoint(Inst, IsDisjoint)

   Sets the disjoint flag for the instruction.

   Only valid for or instructions.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsDisjoint (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetIsDisjoint(LLVMValueRef Inst, LLVMBool IsDisjoint)


.. py:function:: LLVMBuildMalloc(arg0, Ty, Name)

   Memory

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef, LLVMTypeRef Ty, const char * Name)


.. py:function:: LLVMBuildArrayMalloc(arg0, Ty, Val, Name)

   Memory

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Val, const char * Name)


.. py:function:: LLVMBuildMemSet(B, Ptr, Val, Len, Align)

   Creates and inserts a memset to the specified pointer and the
   specified value.

   See:
       llvm::IRRBuilder::CreateMemSet()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Len (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Align (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildMemSet(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Val, LLVMValueRef Len, unsigned int Align)


.. py:function:: LLVMBuildMemCpy(B, Dst, DstAlign, Src, SrcAlign, Size)

   Creates and inserts a memcpy between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemCpy()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DstAlign (:py:obj:`~.int`):
           (undocumented)

       Src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SrcAlign (:py:obj:`~.int`):
           (undocumented)

       Size (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildMemCpy(LLVMBuilderRef B, LLVMValueRef Dst, unsigned int DstAlign, LLVMValueRef Src, unsigned int SrcAlign, LLVMValueRef Size)


.. py:function:: LLVMBuildMemMove(B, Dst, DstAlign, Src, SrcAlign, Size)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DstAlign (:py:obj:`~.int`):
           (undocumented)

       Src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SrcAlign (:py:obj:`~.int`):
           (undocumented)

       Size (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildMemMove(LLVMBuilderRef B, LLVMValueRef Dst, unsigned int DstAlign, LLVMValueRef Src, unsigned int SrcAlign, LLVMValueRef Size)


.. py:function:: LLVMBuildAlloca(arg0, Ty, Name)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef, LLVMTypeRef Ty, const char * Name)


.. py:function:: LLVMBuildArrayAlloca(arg0, Ty, Val, Name)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef Val, const char * Name)


.. py:function:: LLVMBuildFree(arg0, PointerVal)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       PointerVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFree(LLVMBuilderRef, LLVMValueRef PointerVal)


.. py:function:: LLVMBuildLoad2(arg0, Ty, PointerVal, Name)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       PointerVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildLoad2(LLVMBuilderRef, LLVMTypeRef Ty, LLVMValueRef PointerVal, const char * Name)


.. py:function:: LLVMBuildStore(arg0, Val, Ptr)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildStore(LLVMBuilderRef, LLVMValueRef Val, LLVMValueRef Ptr)


.. py:function:: LLVMBuildGEP2(B, Ty, Pointer, Indices, NumIndices, Name)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Pointer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Indices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef * Indices, unsigned int NumIndices, const char * Name)


.. py:function:: LLVMBuildInBoundsGEP2(B, Ty, Pointer, Indices, NumIndices, Name)

   Creates and inserts a memmove between the specified pointers.

   See:
       llvm::IRRBuilder::CreateMemMove()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Pointer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Indices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildInBoundsGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef * Indices, unsigned int NumIndices, const char * Name)


.. py:function:: LLVMBuildGEPWithNoWrapFlags(B, Ty, Pointer, Indices, NumIndices, Name, NoWrapFlags)

   Creates a GetElementPtr instruction.

   Similar to LLVMBuildGEP2, but allows
   specifying the no-wrap flags.

   See:
       llvm::IRBuilder::CreateGEP()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Pointer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Indices (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumIndices (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NoWrapFlags (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildGEPWithNoWrapFlags(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, LLVMValueRef * Indices, unsigned int NumIndices, const char * Name, LLVMGEPNoWrapFlags NoWrapFlags)


.. py:function:: LLVMBuildStructGEP2(B, Ty, Pointer, Idx, Name)

   Creates a GetElementPtr instruction.

   Similar to LLVMBuildGEP2, but allows
   specifying the no-wrap flags.

   See:
       llvm::IRBuilder::CreateGEP()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Pointer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Idx (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildStructGEP2(LLVMBuilderRef B, LLVMTypeRef Ty, LLVMValueRef Pointer, unsigned int Idx, const char * Name)


.. py:function:: LLVMBuildGlobalString(B, Str, Name)

   Creates a GetElementPtr instruction.

   Similar to LLVMBuildGEP2, but allows
   specifying the no-wrap flags.

   See:
       llvm::IRBuilder::CreateGEP()

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildGlobalString(LLVMBuilderRef B, const char * Str, const char * Name)


.. py:function:: LLVMBuildGlobalStringPtr(B, Str, Name)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildGlobalStringPtr(LLVMBuilderRef B, const char * Str, const char * Name)


.. py:function:: LLVMGetVolatile(Inst)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetVolatile(LLVMValueRef Inst)


.. py:function:: LLVMSetVolatile(MemoryAccessInst, IsVolatile)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       MemoryAccessInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsVolatile (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetVolatile(LLVMValueRef MemoryAccessInst, LLVMBool IsVolatile)


.. py:function:: LLVMGetWeak(CmpXchgInst)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetWeak(LLVMValueRef CmpXchgInst)


.. py:function:: LLVMSetWeak(CmpXchgInst, IsWeak)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsWeak (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetWeak(LLVMValueRef CmpXchgInst, LLVMBool IsWeak)


.. py:function:: LLVMGetOrdering(MemoryAccessInst)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       MemoryAccessInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMAtomicOrdering`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAtomicOrdering LLVMGetOrdering(LLVMValueRef MemoryAccessInst)


.. py:function:: LLVMSetOrdering(MemoryAccessInst, Ordering)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       MemoryAccessInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetOrdering(LLVMValueRef MemoryAccessInst, LLVMAtomicOrdering Ordering)


.. py:function:: LLVMGetAtomicRMWBinOp(AtomicRMWInst)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       AtomicRMWInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMAtomicRMWBinOp`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAtomicRMWBinOp LLVMGetAtomicRMWBinOp(LLVMValueRef AtomicRMWInst)


.. py:function:: LLVMSetAtomicRMWBinOp(AtomicRMWInst, BinOp)

   Deprecated: Use LLVMBuildGlobalString instead, which has identical behavior.

   Args:
       AtomicRMWInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       BinOp (:py:obj:`~.LLVMAtomicRMWBinOp`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetAtomicRMWBinOp(LLVMValueRef AtomicRMWInst, LLVMAtomicRMWBinOp BinOp)


.. py:function:: LLVMBuildTrunc(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildZExt(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildZExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildSExt(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildFPToUI(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildFPToSI(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildUIToFP(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildSIToFP(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildFPTrunc(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildFPExt(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildPtrToInt(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildIntToPtr(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildBitCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildAddrSpaceCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAddrSpaceCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildZExtOrBitCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildZExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildSExtOrBitCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildTruncOrBitCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildTruncOrBitCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildCast(B, Op, Val, DestTy, Name)

   Casts

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Op (:py:obj:`~.LLVMOpcode`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCast(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildPointerCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildPointerCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildIntCast2(arg0, Val, DestTy, IsSigned, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       IsSigned (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIntCast2(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, LLVMBool IsSigned, const char * Name)


.. py:function:: LLVMBuildFPCast(arg0, Val, DestTy, Name)

   Casts

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFPCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMBuildIntCast(arg0, Val, DestTy, Name)

   Deprecated: This cast is always signed. Use LLVMBuildIntCast2 instead.

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIntCast(LLVMBuilderRef, LLVMValueRef Val, LLVMTypeRef DestTy, const char * Name)


.. py:function:: LLVMGetCastOpcode(Src, SrcIsSigned, DestTy, DestIsSigned)

   Signed cast!

   Args:
       Src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SrcIsSigned (:py:obj:`~.int`):
           (undocumented)

       DestTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DestIsSigned (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMOpcode`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOpcode LLVMGetCastOpcode(LLVMValueRef Src, LLVMBool SrcIsSigned, LLVMTypeRef DestTy, LLVMBool DestIsSigned)


.. py:function:: LLVMBuildICmp(arg0, Op, LHS, RHS, Name)

   Comparisons

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Op (:py:obj:`~.LLVMIntPredicate`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildICmp(LLVMBuilderRef, LLVMIntPredicate Op, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFCmp(arg0, Op, LHS, RHS, Name)

   Comparisons

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Op (:py:obj:`~.LLVMRealPredicate`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef, LLVMRealPredicate Op, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildPhi(arg0, Ty, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildPhi(LLVMBuilderRef, LLVMTypeRef Ty, const char * Name)


.. py:function:: LLVMBuildCall2(arg0, arg1, Fn, Args, NumArgs, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCall2(LLVMBuilderRef, LLVMTypeRef, LLVMValueRef Fn, LLVMValueRef * Args, unsigned int NumArgs, const char * Name)


.. py:function:: LLVMBuildCallWithOperandBundles(arg0, arg1, Fn, Args, NumArgs, Bundles, NumBundles, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Bundles (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumBundles (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildCallWithOperandBundles(LLVMBuilderRef, LLVMTypeRef, LLVMValueRef Fn, LLVMValueRef * Args, unsigned int NumArgs, LLVMOperandBundleRef * Bundles, unsigned int NumBundles, const char * Name)


.. py:function:: LLVMBuildSelect(arg0, If, Then, Else, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       If (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Then (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Else (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildSelect(LLVMBuilderRef, LLVMValueRef If, LLVMValueRef Then, LLVMValueRef Else, const char * Name)


.. py:function:: LLVMBuildVAArg(arg0, List, Ty, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       List (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef, LLVMValueRef List, LLVMTypeRef Ty, const char * Name)


.. py:function:: LLVMBuildExtractElement(arg0, VecVal, Index, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       VecVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef, LLVMValueRef VecVal, LLVMValueRef Index, const char * Name)


.. py:function:: LLVMBuildInsertElement(arg0, VecVal, EltVal, Index, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       VecVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       EltVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef, LLVMValueRef VecVal, LLVMValueRef EltVal, LLVMValueRef Index, const char * Name)


.. py:function:: LLVMBuildShuffleVector(arg0, V1, V2, Mask, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V1 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       V2 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Mask (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef, LLVMValueRef V1, LLVMValueRef V2, LLVMValueRef Mask, const char * Name)


.. py:function:: LLVMBuildExtractValue(arg0, AggVal, Index, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AggVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef, LLVMValueRef AggVal, unsigned int Index, const char * Name)


.. py:function:: LLVMBuildInsertValue(arg0, AggVal, EltVal, Index, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AggVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       EltVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Index (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef, LLVMValueRef AggVal, LLVMValueRef EltVal, unsigned int Index, const char * Name)


.. py:function:: LLVMBuildFreeze(arg0, Val, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFreeze(LLVMBuilderRef, LLVMValueRef Val, const char * Name)


.. py:function:: LLVMBuildIsNull(arg0, Val, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIsNull(LLVMBuilderRef, LLVMValueRef Val, const char * Name)


.. py:function:: LLVMBuildIsNotNull(arg0, Val, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildIsNotNull(LLVMBuilderRef, LLVMValueRef Val, const char * Name)


.. py:function:: LLVMBuildPtrDiff2(arg0, ElemTy, LHS, RHS, Name)

   Miscellaneous instructions

   Args:
       arg0 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ElemTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       LHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       RHS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildPtrDiff2(LLVMBuilderRef, LLVMTypeRef ElemTy, LLVMValueRef LHS, LLVMValueRef RHS, const char * Name)


.. py:function:: LLVMBuildFence(B, ordering, singleThread, Name)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       singleThread (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFence(LLVMBuilderRef B, LLVMAtomicOrdering ordering, LLVMBool singleThread, const char * Name)


.. py:function:: LLVMBuildFenceSyncScope(B, ordering, SSID, Name)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       SSID (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildFenceSyncScope(LLVMBuilderRef B, LLVMAtomicOrdering ordering, unsigned int SSID, const char * Name)


.. py:function:: LLVMBuildAtomicRMW(B, op, PTR, Val, ordering, singleThread)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       op (:py:obj:`~.LLVMAtomicRMWBinOp`):
           (undocumented)

       PTR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       singleThread (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAtomicRMW(LLVMBuilderRef B, LLVMAtomicRMWBinOp op, LLVMValueRef PTR, LLVMValueRef Val, LLVMAtomicOrdering ordering, LLVMBool singleThread)


.. py:function:: LLVMBuildAtomicRMWSyncScope(B, op, PTR, Val, ordering, SSID)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       op (:py:obj:`~.LLVMAtomicRMWBinOp`):
           (undocumented)

       PTR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       SSID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAtomicRMWSyncScope(LLVMBuilderRef B, LLVMAtomicRMWBinOp op, LLVMValueRef PTR, LLVMValueRef Val, LLVMAtomicOrdering ordering, unsigned int SSID)


.. py:function:: LLVMBuildAtomicCmpXchg(B, Ptr, Cmp, New, SuccessOrdering, FailureOrdering, SingleThread)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Cmp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       New (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SuccessOrdering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       FailureOrdering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       SingleThread (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAtomicCmpXchg(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Cmp, LLVMValueRef New, LLVMAtomicOrdering SuccessOrdering, LLVMAtomicOrdering FailureOrdering, LLVMBool SingleThread)


.. py:function:: LLVMBuildAtomicCmpXchgSyncScope(B, Ptr, Cmp, New, SuccessOrdering, FailureOrdering, SSID)

   Miscellaneous instructions

   Args:
       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Cmp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       New (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SuccessOrdering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       FailureOrdering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

       SSID (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMValueRef LLVMBuildAtomicCmpXchgSyncScope(LLVMBuilderRef B, LLVMValueRef Ptr, LLVMValueRef Cmp, LLVMValueRef New, LLVMAtomicOrdering SuccessOrdering, LLVMAtomicOrdering FailureOrdering, unsigned int SSID)


.. py:function:: LLVMGetNumMaskElements(ShuffleVectorInst)

   Get the number of elements in the mask of a ShuffleVector instruction.

   Args:
       ShuffleVectorInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetNumMaskElements(LLVMValueRef ShuffleVectorInst)


.. py:function:: LLVMGetUndefMaskElem()

   a constant that specifies that the result of a ``ShuffleVectorInst``
   is undefined.

   Returns:
       :py:obj:`~.int`:

   .. rubric:: C signature

   .. code-block:: c

       int LLVMGetUndefMaskElem()


.. py:function:: LLVMGetMaskValue(ShuffleVectorInst, Elt)

   Get the mask value at position Elt in the mask of a ShuffleVector
   instruction.

   \Returns the result of ``LLVMGetUndefMaskElem()`` if the mask value is
   poison at that position.

   Args:
       ShuffleVectorInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Elt (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMGetMaskValue(LLVMValueRef ShuffleVectorInst, unsigned int Elt)


.. py:function:: LLVMIsAtomicSingleThread(AtomicInst)

   Get the mask value at position Elt in the mask of a ShuffleVector
   instruction.

   \Returns the result of ``LLVMGetUndefMaskElem()`` if the mask value is
   poison at that position.

   Args:
       AtomicInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsAtomicSingleThread(LLVMValueRef AtomicInst)


.. py:function:: LLVMSetAtomicSingleThread(AtomicInst, SingleThread)

   Get the mask value at position Elt in the mask of a ShuffleVector
   instruction.

   \Returns the result of ``LLVMGetUndefMaskElem()`` if the mask value is
   poison at that position.

   Args:
       AtomicInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SingleThread (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetAtomicSingleThread(LLVMValueRef AtomicInst, LLVMBool SingleThread)


.. py:function:: LLVMIsAtomic(Inst)

   Returns whether an instruction is an atomic instruction, e.g., atomicrmw,
   cmpxchg, fence, or loads and stores with atomic ordering.

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsAtomic(LLVMValueRef Inst)


.. py:function:: LLVMGetAtomicSyncScopeID(AtomicInst)

   Returns the synchronization scope ID of an atomic instruction.

   Args:
       AtomicInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetAtomicSyncScopeID(LLVMValueRef AtomicInst)


.. py:function:: LLVMSetAtomicSyncScopeID(AtomicInst, SSID)

   Sets the synchronization scope ID of an atomic instruction.

   Args:
       AtomicInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SSID (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetAtomicSyncScopeID(LLVMValueRef AtomicInst, unsigned int SSID)


.. py:function:: LLVMGetCmpXchgSuccessOrdering(CmpXchgInst)

   Sets the synchronization scope ID of an atomic instruction.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMAtomicOrdering`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAtomicOrdering LLVMGetCmpXchgSuccessOrdering(LLVMValueRef CmpXchgInst)


.. py:function:: LLVMSetCmpXchgSuccessOrdering(CmpXchgInst, Ordering)

   Sets the synchronization scope ID of an atomic instruction.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCmpXchgSuccessOrdering(LLVMValueRef CmpXchgInst, LLVMAtomicOrdering Ordering)


.. py:function:: LLVMGetCmpXchgFailureOrdering(CmpXchgInst)

   Sets the synchronization scope ID of an atomic instruction.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMAtomicOrdering`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMAtomicOrdering LLVMGetCmpXchgFailureOrdering(LLVMValueRef CmpXchgInst)


.. py:function:: LLVMSetCmpXchgFailureOrdering(CmpXchgInst, Ordering)

   Sets the synchronization scope ID of an atomic instruction.

   Args:
       CmpXchgInst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Ordering (:py:obj:`~.LLVMAtomicOrdering`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetCmpXchgFailureOrdering(LLVMValueRef CmpXchgInst, LLVMAtomicOrdering Ordering)


.. py:function:: LLVMCreateModuleProviderForExistingModule(M)

   Changes the type of M so it can be passed to FunctionPassManagers and the
   JIT.

   They take ModuleProviders for historical reasons.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMModuleProviderRef LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M)


.. py:function:: LLVMDisposeModuleProvider(M)

   Destroys the module M.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeModuleProvider(LLVMModuleProviderRef M)


.. py:function:: LLVMCreateMemoryBufferWithContentsOfFile(Path)

   (No short description, might be part of a group.)

   Args:
       Path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutMemBuf (:py:obj:`~.LLVMOpaqueMemoryBuffer`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateMemoryBufferWithContentsOfFile(const char * Path, LLVMMemoryBufferRef * OutMemBuf, char ** OutMessage)


.. py:function:: LLVMCreateMemoryBufferWithSTDIN()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutMemBuf (:py:obj:`~.LLVMOpaqueMemoryBuffer`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef * OutMemBuf, char ** OutMessage)


.. py:function:: LLVMCreateMemoryBufferWithMemoryRange(InputData, InputDataLength, BufferName, RequiresNullTerminator)

   (No short description, might be part of a group.)

   Args:
       InputData (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       InputDataLength (:py:obj:`~.int`):
           (undocumented)

       BufferName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       RequiresNullTerminator (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(const char * InputData, size_t InputDataLength, const char * BufferName, LLVMBool RequiresNullTerminator)


.. py:function:: LLVMCreateMemoryBufferWithMemoryRangeCopy(InputData, InputDataLength, BufferName)

   (No short description, might be part of a group.)

   Args:
       InputData (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       InputDataLength (:py:obj:`~.int`):
           (undocumented)

       BufferName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRangeCopy(const char * InputData, size_t InputDataLength, const char * BufferName)


.. py:function:: LLVMGetBufferStart(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetBufferStart(LLVMMemoryBufferRef MemBuf)


.. py:function:: LLVMGetBufferSize(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       size_t LLVMGetBufferSize(LLVMMemoryBufferRef MemBuf)


.. py:function:: LLVMDisposeMemoryBuffer(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf)


.. py:function:: LLVMCreatePassManager()

   Constructs a new whole-module pass pipeline.

   This type of pipeline is
   suitable for link-time optimization and whole-module transformations.

   See:
       llvm::PassManager::PassManager

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMPassManagerRef LLVMCreatePassManager()


.. py:function:: LLVMCreateFunctionPassManagerForModule(M)

   Constructs a new function-by-function pass pipeline over the module
   provider.

   It does not take ownership of the module provider. This type of
   pipeline is suitable for code generation and JIT compilation tasks.

   See:
       llvm::FunctionPassManager::FunctionPassManager

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMPassManagerRef LLVMCreateFunctionPassManagerForModule(LLVMModuleRef M)


.. py:function:: LLVMCreateFunctionPassManager(MP)

   Deprecated: Use LLVMCreateFunctionPassManagerForModule instead.

   Args:
       MP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMPassManagerRef LLVMCreateFunctionPassManager(LLVMModuleProviderRef MP)


.. py:function:: LLVMRunPassManager(PM, M)

   Initializes, executes on the provided module, and finalizes all of the
   passes scheduled in the pass manager.

   Returns 1 if any of the passes
   modified the module, 0 otherwise.

   See:
       llvm::PassManager::run(:py:obj:`~.Module`&)

   Args:
       PM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMRunPassManager(LLVMPassManagerRef PM, LLVMModuleRef M)


.. py:function:: LLVMInitializeFunctionPassManager(FPM)

   Initializes all of the function passes scheduled in the function pass
   manager.

   Returns 1 if any of the passes modified the module, 0 otherwise.

   See:
       llvm::FunctionPassManager::doInitialization

   Args:
       FPM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMInitializeFunctionPassManager(LLVMPassManagerRef FPM)


.. py:function:: LLVMRunFunctionPassManager(FPM, F)

   Executes all of the function passes scheduled in the function pass manager
   on the provided function.

   Returns 1 if any of the passes modified the
   function, false otherwise.

   See:
       llvm::FunctionPassManager::run(:py:obj:`~.Function`&)

   Args:
       FPM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMRunFunctionPassManager(LLVMPassManagerRef FPM, LLVMValueRef F)


.. py:function:: LLVMFinalizeFunctionPassManager(FPM)

   Finalizes all of the function passes scheduled in the function pass
   manager.

   Returns 1 if any of the passes modified the module, 0 otherwise.

   See:
       llvm::FunctionPassManager::doFinalization

   Args:
       FPM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMFinalizeFunctionPassManager(LLVMPassManagerRef FPM)


.. py:function:: LLVMDisposePassManager(PM)

   Frees the memory of a pass pipeline.

   For function pipelines, does not free
   the module provider.

   See:
       llvm::PassManagerBase::~:py:obj:`~.PassManagerBase`.

   Args:
       PM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposePassManager(LLVMPassManagerRef PM)


.. py:function:: LLVMStartMultithreaded()

   Deprecated: Multi-threading can only be enabled/disabled with the compile
   time define LLVM_ENABLE_THREADS.

   This function always returns
   LLVMIsMultithreaded().

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMStartMultithreaded()


.. py:function:: LLVMStopMultithreaded()

   Deprecated: Multi-threading can only be enabled/disabled with the compile
   time define LLVM_ENABLE_THREADS.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMStopMultithreaded()


.. py:function:: LLVMIsMultithreaded()

   Check whether LLVM is executing in thread-safe mode or not.

   See:
       llvm::llvm_is_multithreaded

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsMultithreaded()


