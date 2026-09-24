rocm.bindings.llvm.c.orcee
==========================

.. py:module:: rocm.bindings.llvm.c.orcee


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.orcee.LLVMMemoryManagerCreateContextCallback
   rocm.bindings.llvm.c.orcee.LLVMMemoryManagerNotifyTerminatingCallback


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.orcee.has_symbol
   rocm.bindings.llvm.c.orcee.LLVMOrcCreateObjectLinkingLayerWithInProcessMemoryManager
   rocm.bindings.llvm.c.orcee.LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager
   rocm.bindings.llvm.c.orcee.LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc
   rocm.bindings.llvm.c.orcee.LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks
   rocm.bindings.llvm.c.orcee.LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMMemoryManagerCreateContextCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: LLVMMemoryManagerNotifyTerminatingCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: LLVMOrcCreateObjectLinkingLayerWithInProcessMemoryManager(Result, ES)

   Create a ObjectLinkingLayer instance using the standard JITLink
   InProcessMemoryManager for memory management.

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ES (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateObjectLinkingLayerWithInProcessMemoryManager(LLVMOrcObjectLayerRef * Result, LLVMOrcExecutionSessionRef ES)


.. py:function:: LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(ES)

   Create a RTDyldObjectLinkingLayer instance using the standard
   SectionMemoryManager for memory management.

   Args:
       ES (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcObjectLayerRef LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(LLVMOrcExecutionSessionRef ES)


.. py:function:: LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc(ES, ReserveAlloc)

   Create a RTDyldObjectLinkingLayer instance using the standard
   SectionMemoryManager for memory management.

   If ReserveAlloc is true then
   a contiguous range of memory will be reserved for each object file.

   Args:
       ES (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ReserveAlloc (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcObjectLayerRef LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc(LLVMOrcExecutionSessionRef ES, LLVMBool ReserveAlloc)


.. py:function:: LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(ES, CreateContextCtx, CreateContext, NotifyTerminating, AllocateCodeSection, AllocateDataSection, FinalizeMemory, Destroy)

   Create a RTDyldObjectLinkingLayer instance using MCJIT-memory-manager-like
   callbacks.

   This is intended to simplify transitions for existing MCJIT clients. The
   callbacks used are similar (but not identical) to the callbacks for
   LLVMCreateSimpleMCJITMemoryManager: Unlike MCJIT, RTDyldObjectLinkingLayer
   will create a new memory manager for each object linked by calling the given
   CreateContext callback. This allows for code removal by destroying each
   allocator individually. Every allocator will be destroyed (if it has not been
   already) at RTDyldObjectLinkingLayer destruction time, and the
   NotifyTerminating callback will be called to indicate that no further
   allocation contexts will be created.

   To implement MCJIT-like behavior clients can implement CreateContext,
   NotifyTerminating, and Destroy as:

     void *CreateContext(void *CtxCtx) { return CtxCtx; }
     void NotifyTerminating(void *CtxCtx) { MyOriginalDestroy(CtxCtx); }
     void Destroy(void *Ctx) { }

   This scheme simply reuses the CreateContextCtx pointer as the one-and-only
   allocation context.

   Args:
       ES (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CreateContextCtx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       CreateContext (:py:obj:`~.LLVMMemoryManagerCreateContextCallback`/:py:obj:`~.object`):
           (undocumented)

       NotifyTerminating (:py:obj:`~.LLVMMemoryManagerNotifyTerminatingCallback`/:py:obj:`~.object`):
           (undocumented)

       AllocateCodeSection (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       AllocateDataSection (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       FinalizeMemory (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Destroy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcObjectLayerRef LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(LLVMOrcExecutionSessionRef ES, void * CreateContextCtx, LLVMMemoryManagerCreateContextCallback CreateContext, LLVMMemoryManagerNotifyTerminatingCallback NotifyTerminating, LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection, LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection, LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory, LLVMMemoryManagerDestroyCallback Destroy)


.. py:function:: LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(RTDyldObjLinkingLayer, Listener)

   Add the given listener to the given RTDyldObjectLinkingLayer.

   Note: Layer must be an RTDyldObjectLinkingLayer instance or
   behavior is undefined.

   Args:
       RTDyldObjLinkingLayer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Listener (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(LLVMOrcObjectLayerRef RTDyldObjLinkingLayer, LLVMJITEventListenerRef Listener)


