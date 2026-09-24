rocm.bindings.llvm.c.types
==========================

.. py:module:: rocm.bindings.llvm.c.types


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.types.LLVMMemoryBufferRef
   rocm.bindings.llvm.c.types.LLVMContextRef
   rocm.bindings.llvm.c.types.LLVMModuleRef
   rocm.bindings.llvm.c.types.LLVMTypeRef
   rocm.bindings.llvm.c.types.LLVMValueRef
   rocm.bindings.llvm.c.types.LLVMBasicBlockRef
   rocm.bindings.llvm.c.types.LLVMMetadataRef
   rocm.bindings.llvm.c.types.LLVMNamedMDNodeRef
   rocm.bindings.llvm.c.types.LLVMValueMetadataEntry
   rocm.bindings.llvm.c.types.LLVMBuilderRef
   rocm.bindings.llvm.c.types.LLVMDIBuilderRef
   rocm.bindings.llvm.c.types.LLVMModuleProviderRef
   rocm.bindings.llvm.c.types.LLVMPassManagerRef
   rocm.bindings.llvm.c.types.LLVMUseRef
   rocm.bindings.llvm.c.types.LLVMOperandBundleRef
   rocm.bindings.llvm.c.types.LLVMAttributeRef
   rocm.bindings.llvm.c.types.LLVMDiagnosticInfoRef
   rocm.bindings.llvm.c.types.LLVMComdatRef
   rocm.bindings.llvm.c.types.LLVMModuleFlagEntry
   rocm.bindings.llvm.c.types.LLVMJITEventListenerRef
   rocm.bindings.llvm.c.types.LLVMBinaryRef
   rocm.bindings.llvm.c.types.LLVMDbgRecordRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.types.LLVMOpaqueMemoryBuffer
   rocm.bindings.llvm.c.types.LLVMOpaqueContext
   rocm.bindings.llvm.c.types.LLVMOpaqueModule
   rocm.bindings.llvm.c.types.LLVMOpaqueType
   rocm.bindings.llvm.c.types.LLVMOpaqueValue
   rocm.bindings.llvm.c.types.LLVMOpaqueBasicBlock
   rocm.bindings.llvm.c.types.LLVMOpaqueMetadata
   rocm.bindings.llvm.c.types.LLVMOpaqueNamedMDNode
   rocm.bindings.llvm.c.types.LLVMOpaqueValueMetadataEntry
   rocm.bindings.llvm.c.types.LLVMOpaqueBuilder
   rocm.bindings.llvm.c.types.LLVMOpaqueDIBuilder
   rocm.bindings.llvm.c.types.LLVMOpaqueModuleProvider
   rocm.bindings.llvm.c.types.LLVMOpaquePassManager
   rocm.bindings.llvm.c.types.LLVMOpaqueUse
   rocm.bindings.llvm.c.types.LLVMOpaqueOperandBundle
   rocm.bindings.llvm.c.types.LLVMOpaqueAttributeRef
   rocm.bindings.llvm.c.types.LLVMOpaqueDiagnosticInfo
   rocm.bindings.llvm.c.types.LLVMComdat
   rocm.bindings.llvm.c.types.LLVMOpaqueModuleFlagEntry
   rocm.bindings.llvm.c.types.LLVMOpaqueJITEventListener
   rocm.bindings.llvm.c.types.LLVMOpaqueBinary
   rocm.bindings.llvm.c.types.LLVMOpaqueDbgRecord


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.types.has_symbol


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpaqueMemoryBuffer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMMemoryBufferRef

.. py:class:: LLVMOpaqueContext(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMContextRef

.. py:class:: LLVMOpaqueModule(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMModuleRef

.. py:class:: LLVMOpaqueType(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTypeRef

.. py:class:: LLVMOpaqueValue(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMValueRef

.. py:class:: LLVMOpaqueBasicBlock(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMBasicBlockRef

.. py:class:: LLVMOpaqueMetadata(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMMetadataRef

.. py:class:: LLVMOpaqueNamedMDNode(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMNamedMDNodeRef

.. py:class:: LLVMOpaqueValueMetadataEntry(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMValueMetadataEntry

.. py:class:: LLVMOpaqueBuilder(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMBuilderRef

.. py:class:: LLVMOpaqueDIBuilder(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMDIBuilderRef

.. py:class:: LLVMOpaqueModuleProvider(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMModuleProviderRef

.. py:class:: LLVMOpaquePassManager(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMPassManagerRef

.. py:class:: LLVMOpaqueUse(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMUseRef

.. py:class:: LLVMOpaqueOperandBundle(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOperandBundleRef

.. py:class:: LLVMOpaqueAttributeRef(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMAttributeRef

.. py:class:: LLVMOpaqueDiagnosticInfo(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMDiagnosticInfoRef

.. py:class:: LLVMComdat(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMComdatRef

.. py:class:: LLVMOpaqueModuleFlagEntry(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMModuleFlagEntry

.. py:class:: LLVMOpaqueJITEventListener(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMJITEventListenerRef

.. py:class:: LLVMOpaqueBinary(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMBinaryRef

.. py:class:: LLVMOpaqueDbgRecord(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMDbgRecordRef

