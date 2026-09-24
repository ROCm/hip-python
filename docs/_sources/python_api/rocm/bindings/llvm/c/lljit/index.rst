rocm.bindings.llvm.c.lljit
==========================

.. py:module:: rocm.bindings.llvm.c.lljit


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITBuilderRef
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction
   rocm.bindings.llvm.c.lljit.LLVMOrcOpaqueLLJITBuilder
   rocm.bindings.llvm.c.lljit.LLVMOrcOpaqueLLJIT


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.lljit.has_symbol
   rocm.bindings.llvm.c.lljit.LLVMOrcCreateLLJITBuilder
   rocm.bindings.llvm.c.lljit.LLVMOrcDisposeLLJITBuilder
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator
   rocm.bindings.llvm.c.lljit.LLVMOrcCreateLLJIT
   rocm.bindings.llvm.c.lljit.LLVMOrcDisposeLLJIT
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetExecutionSession
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetMainJITDylib
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetTripleString
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetGlobalPrefix
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITMangleAndIntern
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITAddObjectFile
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITAddObjectFileWithRT
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITAddLLVMIRModule
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITAddLLVMIRModuleWithRT
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITLookup
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetObjLinkingLayer
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetObjTransformLayer
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetIRTransformLayer
   rocm.bindings.llvm.c.lljit.LLVMOrcLLJITGetDataLayoutStr


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A function for constructing an ObjectLinkingLayer instance to be used
   by an LLJIT instance.

   Clients can call LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator to
   set the creator function to use when constructing an LLJIT instance.
   This can be used to override the default linking layer implementation
   that would otherwise be chosen by LLJITBuilder.

   Object linking layers returned by this function will become owned by the
   LLJIT instance. The client is not responsible for managing their lifetimes
   after the function returns.

   FIXME: This method needs to be updated to take a JITLinkMemoryManager
          argument.


.. py:class:: LLVMOrcOpaqueLLJITBuilder(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcLLJITBuilderRef

.. py:class:: LLVMOrcOpaqueLLJIT(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcLLJITRef

.. py:function:: LLVMOrcCreateLLJITBuilder()

   Create an LLVMOrcLLJITBuilder.

   The client owns the resulting LLJITBuilder and should dispose of it using
   LLVMOrcDisposeLLJITBuilder once they are done with it.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcLLJITBuilderRef LLVMOrcCreateLLJITBuilder()


.. py:function:: LLVMOrcDisposeLLJITBuilder(Builder)

   Dispose of an LLVMOrcLLJITBuilderRef.

   This should only be called if ownership
   has not been passed to LLVMOrcCreateLLJIT (e.g. because some error prevented
   that function from being called).

   Args:
       Builder (:py:obj:`~.LLVMOrcOpaqueLLJITBuilder`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeLLJITBuilder(LLVMOrcLLJITBuilderRef Builder)


.. py:function:: LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(Builder, JTMB)

   Set the JITTargetMachineBuilder to be used when constructing the LLJIT
   instance.

   Calling this function is optional: if it is not called then the
   LLJITBuilder will use JITTargeTMachineBuilder::detectHost to construct a
   JITTargetMachineBuilder.

   This function takes ownership of the JTMB argument: clients should not
   dispose of the JITTargetMachineBuilder after calling this function.

   Args:
       Builder (:py:obj:`~.LLVMOrcOpaqueLLJITBuilder`/:py:obj:`~.object`):
           (undocumented)

       JTMB (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(LLVMOrcLLJITBuilderRef Builder, LLVMOrcJITTargetMachineBuilderRef JTMB)


.. py:function:: LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(Builder, F, Ctx)

   Set an ObjectLinkingLayer creator function for this LLJIT instance.

   Args:
       Builder (:py:obj:`~.LLVMOrcOpaqueLLJITBuilder`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(LLVMOrcLLJITBuilderRef Builder, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction F, void * Ctx)


.. py:function:: LLVMOrcCreateLLJIT(Result, Builder)

   Create an LLJIT instance from an LLJITBuilder.

   This operation takes ownership of the Builder argument: clients should not
   dispose of the builder after calling this function (even if the function
   returns an error). If a null Builder argument is provided then a
   default-constructed LLJITBuilder will be used.

   On success the resulting LLJIT instance is uniquely owned by the client and
   automatically manages the memory of all JIT'd code and all modules that are
   transferred to it (e.g. via LLVMOrcLLJITAddLLVMIRModule). Disposing of the
   LLJIT instance will free all memory managed by the JIT, including JIT'd code
   and not-yet compiled modules.

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Builder (:py:obj:`~.LLVMOrcOpaqueLLJITBuilder`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateLLJIT(LLVMOrcLLJITRef * Result, LLVMOrcLLJITBuilderRef Builder)


.. py:function:: LLVMOrcDisposeLLJIT(J)

   Dispose of an LLJIT instance.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcDisposeLLJIT(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetExecutionSession(J)

   Get a reference to the ExecutionSession for this LLJIT instance.

   The ExecutionSession is owned by the LLJIT instance. The client is not
   responsible for managing its memory.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcExecutionSessionRef LLVMOrcLLJITGetExecutionSession(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetMainJITDylib(J)

   Return a reference to the Main JITDylib.

   The JITDylib is owned by the LLJIT instance. The client is not responsible
   for managing its memory.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcJITDylibRef LLVMOrcLLJITGetMainJITDylib(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetTripleString(J)

   Return the target triple for this LLJIT instance.

   This string is owned by
   the LLJIT instance and should not be freed by the client.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMOrcLLJITGetTripleString(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetGlobalPrefix(J)

   Returns the global prefix character according to the LLJIT's DataLayout.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char LLVMOrcLLJITGetGlobalPrefix(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITMangleAndIntern(J, UnmangledName)

   Mangles the given string according to the LLJIT instance's DataLayout, then
   interns the result in the SymbolStringPool and returns a reference to the
   pool entry.

   Clients should call LLVMOrcReleaseSymbolStringPoolEntry to
   decrement the ref-count on the pool entry once they are finished with this
   value.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       UnmangledName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcSymbolStringPoolEntryRef LLVMOrcLLJITMangleAndIntern(LLVMOrcLLJITRef J, const char * UnmangledName)


.. py:function:: LLVMOrcLLJITAddObjectFile(J, JD, ObjBuffer)

   Add a buffer representing an object file to the given JITDylib in the given
   LLJIT instance.

   This operation transfers ownership of the buffer to the
   LLJIT instance. The buffer should not be disposed of or referenced once this
   function returns.

   Resources associated with the given object will be tracked by the given
   JITDylib's default resource tracker.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       JD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITAddObjectFile(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)


.. py:function:: LLVMOrcLLJITAddObjectFileWithRT(J, RT, ObjBuffer)

   Add a buffer representing an object file to the given ResourceTracker's
   JITDylib in the given LLJIT instance.

   This operation transfers ownership of
   the buffer to the LLJIT instance. The buffer should not be disposed of or
   referenced once this function returns.

   Resources associated with the given object will be tracked by ResourceTracker
   RT.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       RT (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITAddObjectFileWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)


.. py:function:: LLVMOrcLLJITAddLLVMIRModule(J, JD, TSM)

   Add an IR module to the given JITDylib in the given LLJIT instance.

   This
   operation transfers ownership of the TSM argument to the LLJIT instance.
   The TSM argument should not be disposed of or referenced once this
   function returns.

   Resources associated with the given Module will be tracked by the given
   JITDylib's default resource tracker.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       JD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TSM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITAddLLVMIRModule(LLVMOrcLLJITRef J, LLVMOrcJITDylibRef JD, LLVMOrcThreadSafeModuleRef TSM)


.. py:function:: LLVMOrcLLJITAddLLVMIRModuleWithRT(J, JD, TSM)

   Add an IR module to the given ResourceTracker's JITDylib in the given LLJIT
   instance.

   This operation transfers ownership of the TSM argument to the LLJIT
   instance. The TSM argument should not be disposed of or referenced once this
   function returns.

   Resources associated with the given Module will be tracked by ResourceTracker
   RT.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       JD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TSM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITAddLLVMIRModuleWithRT(LLVMOrcLLJITRef J, LLVMOrcResourceTrackerRef JD, LLVMOrcThreadSafeModuleRef TSM)


.. py:function:: LLVMOrcLLJITLookup(J, Result, Name)

   Look up the given symbol in the main JITDylib of the given LLJIT instance.

   This operation does not take ownership of the Name argument.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

       Result (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITLookup(LLVMOrcLLJITRef J, LLVMOrcExecutorAddress * Result, const char * Name)


.. py:function:: LLVMOrcLLJITGetObjLinkingLayer(J)

   Returns a non-owning reference to the LLJIT instance's object linking layer.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcObjectLayerRef LLVMOrcLLJITGetObjLinkingLayer(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetObjTransformLayer(J)

   Returns a non-owning reference to the LLJIT instance's object linking layer.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcObjectTransformLayerRef LLVMOrcLLJITGetObjTransformLayer(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetIRTransformLayer(J)

   Returns a non-owning reference to the LLJIT instance's IR transform layer.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcIRTransformLayerRef LLVMOrcLLJITGetIRTransformLayer(LLVMOrcLLJITRef J)


.. py:function:: LLVMOrcLLJITGetDataLayoutStr(J)

   Get the LLJIT instance's default data layout string.

   This string is owned by the LLJIT instance and does not need to be freed
   by the caller.

   Args:
       J (:py:obj:`~.LLVMOrcOpaqueLLJIT`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMOrcLLJITGetDataLayoutStr(LLVMOrcLLJITRef J)


