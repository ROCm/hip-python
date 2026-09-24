rocm.bindings.llvm.c.transforms.passbuilder
===========================================

.. py:module:: rocm.bindings.llvm.c.transforms.passbuilder


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.transforms.passbuilder.LLVMOpaquePassBuilderOptions


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.transforms.passbuilder.has_symbol
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMRunPasses
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMRunPassesOnFunction
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMCreatePassBuilderOptions
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetVerifyEach
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetDebugLogging
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetAAPipeline
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetLoopInterleaving
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetLoopVectorization
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetSLPVectorization
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetLoopUnrolling
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetLicmMssaOptCap
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetCallGraphProfile
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetMergeFunctions
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMPassBuilderOptionsSetInlinerThreshold
   rocm.bindings.llvm.c.transforms.passbuilder.LLVMDisposePassBuilderOptions


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpaquePassBuilderOptions(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMPassBuilderOptionsRef

.. py:function:: LLVMRunPasses(M, Passes, TM, Options)

   Construct and run a set of passes over a module

   This function takes a string with the passes that should be used. The format
   of this string is the same as opt's -passes argument for the new pass
   manager. Individual passes may be specified, separated by commas. Full
   pipelines may also be invoked using `default<O3>` and friends. See opt for
   full reference of the Passes format.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Passes (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       TM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMRunPasses(LLVMModuleRef M, const char * Passes, LLVMTargetMachineRef TM, LLVMPassBuilderOptionsRef Options)


.. py:function:: LLVMRunPassesOnFunction(F, Passes, TM, Options)

   Construct and run a set of passes over a function.

   This function behaves the same as LLVMRunPasses, but operates on a single
   function instead of an entire module.

   Args:
       F (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Passes (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       TM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMRunPassesOnFunction(LLVMValueRef F, const char * Passes, LLVMTargetMachineRef TM, LLVMPassBuilderOptionsRef Options)


.. py:function:: LLVMCreatePassBuilderOptions()

   Create a new set of options for a PassBuilder

   Ownership of the returned instance is given to the client, and they are
   responsible for it. The client should call LLVMDisposePassBuilderOptions
   to free the pass builder options.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMPassBuilderOptionsRef LLVMCreatePassBuilderOptions()


.. py:function:: LLVMPassBuilderOptionsSetVerifyEach(Options, VerifyEach)

   Toggle adding the VerifierPass for the PassBuilder, ensuring all functions
   inside the module is valid.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       VerifyEach (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetVerifyEach(LLVMPassBuilderOptionsRef Options, LLVMBool VerifyEach)


.. py:function:: LLVMPassBuilderOptionsSetDebugLogging(Options, DebugLogging)

   Toggle debug logging when running the PassBuilder

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       DebugLogging (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetDebugLogging(LLVMPassBuilderOptionsRef Options, LLVMBool DebugLogging)


.. py:function:: LLVMPassBuilderOptionsSetAAPipeline(Options, AAPipeline)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       AAPipeline (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetAAPipeline(LLVMPassBuilderOptionsRef Options, const char * AAPipeline)


.. py:function:: LLVMPassBuilderOptionsSetLoopInterleaving(Options, LoopInterleaving)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       LoopInterleaving (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetLoopInterleaving(LLVMPassBuilderOptionsRef Options, LLVMBool LoopInterleaving)


.. py:function:: LLVMPassBuilderOptionsSetLoopVectorization(Options, LoopVectorization)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       LoopVectorization (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetLoopVectorization(LLVMPassBuilderOptionsRef Options, LLVMBool LoopVectorization)


.. py:function:: LLVMPassBuilderOptionsSetSLPVectorization(Options, SLPVectorization)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       SLPVectorization (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetSLPVectorization(LLVMPassBuilderOptionsRef Options, LLVMBool SLPVectorization)


.. py:function:: LLVMPassBuilderOptionsSetLoopUnrolling(Options, LoopUnrolling)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       LoopUnrolling (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetLoopUnrolling(LLVMPassBuilderOptionsRef Options, LLVMBool LoopUnrolling)


.. py:function:: LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(Options, ForgetAllSCEVInLoopUnroll)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       ForgetAllSCEVInLoopUnroll (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll(LLVMPassBuilderOptionsRef Options, LLVMBool ForgetAllSCEVInLoopUnroll)


.. py:function:: LLVMPassBuilderOptionsSetLicmMssaOptCap(Options, LicmMssaOptCap)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       LicmMssaOptCap (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetLicmMssaOptCap(LLVMPassBuilderOptionsRef Options, unsigned int LicmMssaOptCap)


.. py:function:: LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(Options, LicmMssaNoAccForPromotionCap)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       LicmMssaNoAccForPromotionCap (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap(LLVMPassBuilderOptionsRef Options, unsigned int LicmMssaNoAccForPromotionCap)


.. py:function:: LLVMPassBuilderOptionsSetCallGraphProfile(Options, CallGraphProfile)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       CallGraphProfile (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetCallGraphProfile(LLVMPassBuilderOptionsRef Options, LLVMBool CallGraphProfile)


.. py:function:: LLVMPassBuilderOptionsSetMergeFunctions(Options, MergeFunctions)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       MergeFunctions (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetMergeFunctions(LLVMPassBuilderOptionsRef Options, LLVMBool MergeFunctions)


.. py:function:: LLVMPassBuilderOptionsSetInlinerThreshold(Options, Threshold)

   Specify a custom alias analysis pipeline for the PassBuilder to be used
   instead of the default one.

   The string argument is not copied; the caller
   is responsible for ensuring it outlives the PassBuilderOptions instance.

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

       Threshold (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMPassBuilderOptionsSetInlinerThreshold(LLVMPassBuilderOptionsRef Options, int Threshold)


.. py:function:: LLVMDisposePassBuilderOptions(Options)

   Dispose of a heap-allocated PassBuilderOptions instance

   Args:
       Options (:py:obj:`~.LLVMOpaquePassBuilderOptions`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposePassBuilderOptions(LLVMPassBuilderOptionsRef Options)


