rocm.bindings.llvm.c.executionengine
====================================

.. py:module:: rocm.bindings.llvm.c.executionengine


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.executionengine.LLVMGenericValueRef
   rocm.bindings.llvm.c.executionengine.LLVMExecutionEngineRef
   rocm.bindings.llvm.c.executionengine.LLVMMCJITMemoryManagerRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.executionengine.LLVMOpaqueGenericValue
   rocm.bindings.llvm.c.executionengine.LLVMOpaqueExecutionEngine
   rocm.bindings.llvm.c.executionengine.LLVMOpaqueMCJITMemoryManager
   rocm.bindings.llvm.c.executionengine.LLVMMCJITCompilerOptions
   rocm.bindings.llvm.c.executionengine.LLVMMemoryManagerAllocateCodeSectionCallback
   rocm.bindings.llvm.c.executionengine.LLVMMemoryManagerAllocateDataSectionCallback
   rocm.bindings.llvm.c.executionengine.LLVMMemoryManagerFinalizeMemoryCallback
   rocm.bindings.llvm.c.executionengine.LLVMMemoryManagerDestroyCallback


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.executionengine.has_symbol
   rocm.bindings.llvm.c.executionengine.LLVMLinkInMCJIT
   rocm.bindings.llvm.c.executionengine.LLVMLinkInInterpreter
   rocm.bindings.llvm.c.executionengine.LLVMCreateGenericValueOfInt
   rocm.bindings.llvm.c.executionengine.LLVMCreateGenericValueOfPointer
   rocm.bindings.llvm.c.executionengine.LLVMCreateGenericValueOfFloat
   rocm.bindings.llvm.c.executionengine.LLVMGenericValueIntWidth
   rocm.bindings.llvm.c.executionengine.LLVMGenericValueToInt
   rocm.bindings.llvm.c.executionengine.LLVMGenericValueToPointer
   rocm.bindings.llvm.c.executionengine.LLVMGenericValueToFloat
   rocm.bindings.llvm.c.executionengine.LLVMDisposeGenericValue
   rocm.bindings.llvm.c.executionengine.LLVMCreateExecutionEngineForModule
   rocm.bindings.llvm.c.executionengine.LLVMCreateInterpreterForModule
   rocm.bindings.llvm.c.executionengine.LLVMCreateJITCompilerForModule
   rocm.bindings.llvm.c.executionengine.LLVMInitializeMCJITCompilerOptions
   rocm.bindings.llvm.c.executionengine.LLVMCreateMCJITCompilerForModule
   rocm.bindings.llvm.c.executionengine.LLVMDisposeExecutionEngine
   rocm.bindings.llvm.c.executionengine.LLVMRunStaticConstructors
   rocm.bindings.llvm.c.executionengine.LLVMRunStaticDestructors
   rocm.bindings.llvm.c.executionengine.LLVMRunFunctionAsMain
   rocm.bindings.llvm.c.executionengine.LLVMRunFunction
   rocm.bindings.llvm.c.executionengine.LLVMFreeMachineCodeForFunction
   rocm.bindings.llvm.c.executionengine.LLVMAddModule
   rocm.bindings.llvm.c.executionengine.LLVMRemoveModule
   rocm.bindings.llvm.c.executionengine.LLVMFindFunction
   rocm.bindings.llvm.c.executionengine.LLVMRecompileAndRelinkFunction
   rocm.bindings.llvm.c.executionengine.LLVMGetExecutionEngineTargetData
   rocm.bindings.llvm.c.executionengine.LLVMGetExecutionEngineTargetMachine
   rocm.bindings.llvm.c.executionengine.LLVMAddGlobalMapping
   rocm.bindings.llvm.c.executionengine.LLVMGetPointerToGlobal
   rocm.bindings.llvm.c.executionengine.LLVMGetGlobalValueAddress
   rocm.bindings.llvm.c.executionengine.LLVMGetFunctionAddress
   rocm.bindings.llvm.c.executionengine.LLVMExecutionEngineGetErrMsg
   rocm.bindings.llvm.c.executionengine.LLVMCreateSimpleMCJITMemoryManager
   rocm.bindings.llvm.c.executionengine.LLVMDisposeMCJITMemoryManager
   rocm.bindings.llvm.c.executionengine.LLVMCreateGDBRegistrationListener
   rocm.bindings.llvm.c.executionengine.LLVMCreateIntelJITEventListener
   rocm.bindings.llvm.c.executionengine.LLVMCreateOProfileJITEventListener
   rocm.bindings.llvm.c.executionengine.LLVMCreatePerfJITEventListener


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMLinkInMCJIT()

   Empty function used to force the linker to link MCJIT.

   Has no effect when called on a pre-built library (dylib interface).

   .. rubric:: C signature

   .. code-block:: c

       void LLVMLinkInMCJIT()


.. py:function:: LLVMLinkInInterpreter()

   Empty function used to force the linker to link the LLVM interpreter.

   Has no effect when called on a pre-built library (dylib interface).

   .. rubric:: C signature

   .. code-block:: c

       void LLVMLinkInInterpreter()


.. py:class:: LLVMOpaqueGenericValue(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMGenericValueRef

.. py:class:: LLVMOpaqueExecutionEngine(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMExecutionEngineRef

.. py:class:: LLVMOpaqueMCJITMemoryManager(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMMCJITMemoryManagerRef

.. py:class:: LLVMMCJITCompilerOptions(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: OptLevel
      :type:  Any


   .. py:attribute:: CodeModel
      :type:  Any


   .. py:attribute:: NoFramePointerElim
      :type:  Any


   .. py:attribute:: EnableFastISel
      :type:  Any


   .. py:attribute:: MCJMM
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: LLVMCreateGenericValueOfInt(Ty, N, IsSigned)

   ===-- Operations on generic values --------------------------------------===

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       N (:py:obj:`~.int`):
           (undocumented)

       IsSigned (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMGenericValueRef LLVMCreateGenericValueOfInt(LLVMTypeRef Ty, unsigned long long N, LLVMBool IsSigned)


.. py:function:: LLVMCreateGenericValueOfPointer(P)

   ===-- Operations on generic values --------------------------------------===

   Args:
       P (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMGenericValueRef LLVMCreateGenericValueOfPointer(void * P)


.. py:function:: LLVMCreateGenericValueOfFloat(Ty, N)

   ===-- Operations on generic values --------------------------------------===

   Args:
       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       N (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMGenericValueRef LLVMCreateGenericValueOfFloat(LLVMTypeRef Ty, double N)


.. py:function:: LLVMGenericValueIntWidth(GenValRef)

   ===-- Operations on generic values --------------------------------------===

   Args:
       GenValRef (:py:obj:`~.LLVMOpaqueGenericValue`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGenericValueIntWidth(LLVMGenericValueRef GenValRef)


.. py:function:: LLVMGenericValueToInt(GenVal, IsSigned)

   ===-- Operations on generic values --------------------------------------===

   Args:
       GenVal (:py:obj:`~.LLVMOpaqueGenericValue`/:py:obj:`~.object`):
           (undocumented)

       IsSigned (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMGenericValueToInt(LLVMGenericValueRef GenVal, LLVMBool IsSigned)


.. py:function:: LLVMGenericValueToPointer(GenVal)

   ===-- Operations on generic values --------------------------------------===

   Args:
       GenVal (:py:obj:`~.LLVMOpaqueGenericValue`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void * LLVMGenericValueToPointer(LLVMGenericValueRef GenVal)


.. py:function:: LLVMGenericValueToFloat(TyRef, GenVal)

   ===-- Operations on generic values --------------------------------------===

   Args:
       TyRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       GenVal (:py:obj:`~.LLVMOpaqueGenericValue`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.float`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       double LLVMGenericValueToFloat(LLVMTypeRef TyRef, LLVMGenericValueRef GenVal)


.. py:function:: LLVMDisposeGenericValue(GenVal)

   ===-- Operations on generic values --------------------------------------===

   Args:
       GenVal (:py:obj:`~.LLVMOpaqueGenericValue`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeGenericValue(LLVMGenericValueRef GenVal)


.. py:function:: LLVMCreateExecutionEngineForModule(M)

   ===-- Operations on execution engines -----------------------------------===

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutEE (:py:obj:`~.LLVMOpaqueExecutionEngine`):
           (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateExecutionEngineForModule(LLVMExecutionEngineRef * OutEE, LLVMModuleRef M, char ** OutError)


.. py:function:: LLVMCreateInterpreterForModule(M)

   ===-- Operations on execution engines -----------------------------------===

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutInterp (:py:obj:`~.LLVMOpaqueExecutionEngine`):
           (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateInterpreterForModule(LLVMExecutionEngineRef * OutInterp, LLVMModuleRef M, char ** OutError)


.. py:function:: LLVMCreateJITCompilerForModule(M, OptLevel)

   ===-- Operations on execution engines -----------------------------------===

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OptLevel (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutJIT (:py:obj:`~.LLVMOpaqueExecutionEngine`):
           (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateJITCompilerForModule(LLVMExecutionEngineRef * OutJIT, LLVMModuleRef M, unsigned int OptLevel, char ** OutError)


.. py:function:: LLVMInitializeMCJITCompilerOptions(Options, SizeOfOptions)

   ===-- Operations on execution engines -----------------------------------===

   Args:
       Options (:py:obj:`~.LLVMMCJITCompilerOptions`/:py:obj:`~.object`):
           (undocumented)

       SizeOfOptions (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeMCJITCompilerOptions(struct LLVMMCJITCompilerOptions * Options, size_t SizeOfOptions)


.. py:function:: LLVMCreateMCJITCompilerForModule(M, Options, SizeOfOptions)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Options (:py:obj:`~.LLVMMCJITCompilerOptions`/:py:obj:`~.object`):
           (undocumented)

       SizeOfOptions (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutJIT (:py:obj:`~.LLVMOpaqueExecutionEngine`):
           (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMCreateMCJITCompilerForModule(LLVMExecutionEngineRef * OutJIT, LLVMModuleRef M, struct LLVMMCJITCompilerOptions * Options, size_t SizeOfOptions, char ** OutError)


.. py:function:: LLVMDisposeExecutionEngine(EE)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeExecutionEngine(LLVMExecutionEngineRef EE)


.. py:function:: LLVMRunStaticConstructors(EE)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRunStaticConstructors(LLVMExecutionEngineRef EE)


.. py:function:: LLVMRunStaticDestructors(EE)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRunStaticDestructors(LLVMExecutionEngineRef EE)


.. py:function:: LLVMRunFunctionAsMain(EE, F, ArgC, ArgV, EnvP)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ArgC (:py:obj:`~.int`):
           (undocumented)

       ArgV (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       EnvP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMRunFunctionAsMain(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int ArgC, const char *const * ArgV, const char *const * EnvP)


.. py:function:: LLVMRunFunction(EE, F, NumArgs, Args)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumArgs (:py:obj:`~.int`):
           (undocumented)

       Args (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMGenericValueRef LLVMRunFunction(LLVMExecutionEngineRef EE, LLVMValueRef F, unsigned int NumArgs, LLVMGenericValueRef * Args)


.. py:function:: LLVMFreeMachineCodeForFunction(EE, F)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMFreeMachineCodeForFunction(LLVMExecutionEngineRef EE, LLVMValueRef F)


.. py:function:: LLVMAddModule(EE, M)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddModule(LLVMExecutionEngineRef EE, LLVMModuleRef M)


.. py:function:: LLVMRemoveModule(EE, M)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutMod (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMRemoveModule(LLVMExecutionEngineRef EE, LLVMModuleRef M, LLVMModuleRef * OutMod, char ** OutError)


.. py:function:: LLVMFindFunction(EE, Name)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutFn (:py:obj:`~.LLVMOpaqueValue`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMFindFunction(LLVMExecutionEngineRef EE, const char * Name, LLVMValueRef * OutFn)


.. py:function:: LLVMRecompileAndRelinkFunction(EE, Fn)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void * LLVMRecompileAndRelinkFunction(LLVMExecutionEngineRef EE, LLVMValueRef Fn)


.. py:function:: LLVMGetExecutionEngineTargetData(EE)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetDataRef LLVMGetExecutionEngineTargetData(LLVMExecutionEngineRef EE)


.. py:function:: LLVMGetExecutionEngineTargetMachine(EE)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetMachineRef LLVMGetExecutionEngineTargetMachine(LLVMExecutionEngineRef EE)


.. py:function:: LLVMAddGlobalMapping(EE, Global, Addr)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Addr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddGlobalMapping(LLVMExecutionEngineRef EE, LLVMValueRef Global, void * Addr)


.. py:function:: LLVMGetPointerToGlobal(EE, Global)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Global (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void * LLVMGetPointerToGlobal(LLVMExecutionEngineRef EE, LLVMValueRef Global)


.. py:function:: LLVMGetGlobalValueAddress(EE, Name)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetGlobalValueAddress(LLVMExecutionEngineRef EE, const char * Name)


.. py:function:: LLVMGetFunctionAddress(EE, Name)

   Create an MCJIT execution engine for a module, with the given options.

   It is
   the responsibility of the caller to ensure that all fields in Options up to
   the given SizeOfOptions are initialized. It is correct to pass a smaller
   value of SizeOfOptions that omits some fields. The canonical way of using
   this is:

   LLVMMCJITCompilerOptions options;
   LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
   ... fill in those options you care about
   LLVMCreateMCJITCompilerForModule(&jit, mod, &options, sizeof(options),
                                    &error);

   Note that this is also correct, though possibly suboptimal:

   LLVMCreateMCJITCompilerForModule(&jit, mod, 0, 0, &error);

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetFunctionAddress(LLVMExecutionEngineRef EE, const char * Name)


.. py:function:: LLVMExecutionEngineGetErrMsg(EE)

   Returns true on error, false on success.

   If true is returned then the error
   message is copied to OutStr and cleared in the ExecutionEngine instance.

   Args:
       EE (:py:obj:`~.LLVMOpaqueExecutionEngine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutError (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMExecutionEngineGetErrMsg(LLVMExecutionEngineRef EE, char ** OutError)


.. py:class:: LLVMMemoryManagerAllocateCodeSectionCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: LLVMMemoryManagerAllocateDataSectionCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: LLVMMemoryManagerFinalizeMemoryCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: LLVMMemoryManagerDestroyCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: LLVMCreateSimpleMCJITMemoryManager(Opaque, AllocateCodeSection, AllocateDataSection, FinalizeMemory, Destroy)

   Create a simple custom MCJIT memory manager.

   This memory manager can
   intercept allocations in a module-oblivious way. This will return NULL
   if any of the passed functions are NULL.

   Args:
       Opaque (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           An opaque client object to pass back to the callbacks.

       AllocateCodeSection (:py:obj:`~.LLVMMemoryManagerAllocateCodeSectionCallback`/:py:obj:`~.object`):
           Allocate a block of memory for executable code.

       AllocateDataSection (:py:obj:`~.LLVMMemoryManagerAllocateDataSectionCallback`/:py:obj:`~.object`):
           Allocate a block of memory for data.

       FinalizeMemory (:py:obj:`~.LLVMMemoryManagerFinalizeMemoryCallback`/:py:obj:`~.object`):
           Set page permissions and flush cache. Return 0 on
           success, 1 on error.

       Destroy (:py:obj:`~.LLVMMemoryManagerDestroyCallback`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMCJITMemoryManagerRef LLVMCreateSimpleMCJITMemoryManager(void * Opaque, LLVMMemoryManagerAllocateCodeSectionCallback AllocateCodeSection, LLVMMemoryManagerAllocateDataSectionCallback AllocateDataSection, LLVMMemoryManagerFinalizeMemoryCallback FinalizeMemory, LLVMMemoryManagerDestroyCallback Destroy)


.. py:function:: LLVMDisposeMCJITMemoryManager(MM)

   Create a simple custom MCJIT memory manager.

   This memory manager can
   intercept allocations in a module-oblivious way. This will return NULL
   if any of the passed functions are NULL.

   Args:
       MM (:py:obj:`~.LLVMOpaqueMCJITMemoryManager`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeMCJITMemoryManager(LLVMMCJITMemoryManagerRef MM)


.. py:function:: LLVMCreateGDBRegistrationListener()

   ===-- JIT Event Listener functions -------------------------------------===

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMJITEventListenerRef LLVMCreateGDBRegistrationListener()


.. py:function:: LLVMCreateIntelJITEventListener()

   ===-- JIT Event Listener functions -------------------------------------===

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMJITEventListenerRef LLVMCreateIntelJITEventListener()


.. py:function:: LLVMCreateOProfileJITEventListener()

   ===-- JIT Event Listener functions -------------------------------------===

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMJITEventListenerRef LLVMCreateOProfileJITEventListener()


.. py:function:: LLVMCreatePerfJITEventListener()

   ===-- JIT Event Listener functions -------------------------------------===

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMJITEventListenerRef LLVMCreatePerfJITEventListener()


