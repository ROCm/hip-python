rocm.bindings.llvm.c.targetmachine
==================================

.. py:module:: rocm.bindings.llvm.c.targetmachine


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsRef
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineRef
   rocm.bindings.llvm.c.targetmachine.LLVMTargetRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.targetmachine.LLVMOpaqueTargetMachineOptions
   rocm.bindings.llvm.c.targetmachine.LLVMOpaqueTargetMachine
   rocm.bindings.llvm.c.targetmachine.LLVMTarget
   rocm.bindings.llvm.c.targetmachine.LLVMCodeGenOptLevel
   rocm.bindings.llvm.c.targetmachine.LLVMRelocMode
   rocm.bindings.llvm.c.targetmachine.LLVMCodeModel
   rocm.bindings.llvm.c.targetmachine.LLVMCodeGenFileType
   rocm.bindings.llvm.c.targetmachine.LLVMGlobalISelAbortMode


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.targetmachine.has_symbol
   rocm.bindings.llvm.c.targetmachine.LLVMGetFirstTarget
   rocm.bindings.llvm.c.targetmachine.LLVMGetNextTarget
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetFromName
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetFromTriple
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetName
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetDescription
   rocm.bindings.llvm.c.targetmachine.LLVMTargetHasJIT
   rocm.bindings.llvm.c.targetmachine.LLVMTargetHasTargetMachine
   rocm.bindings.llvm.c.targetmachine.LLVMTargetHasAsmBackend
   rocm.bindings.llvm.c.targetmachine.LLVMCreateTargetMachineOptions
   rocm.bindings.llvm.c.targetmachine.LLVMDisposeTargetMachineOptions
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetCPU
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetFeatures
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetABI
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetCodeGenOptLevel
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetRelocMode
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineOptionsSetCodeModel
   rocm.bindings.llvm.c.targetmachine.LLVMCreateTargetMachineWithOptions
   rocm.bindings.llvm.c.targetmachine.LLVMCreateTargetMachine
   rocm.bindings.llvm.c.targetmachine.LLVMDisposeTargetMachine
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetMachineTarget
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetMachineTriple
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetMachineCPU
   rocm.bindings.llvm.c.targetmachine.LLVMGetTargetMachineFeatureString
   rocm.bindings.llvm.c.targetmachine.LLVMCreateTargetDataLayout
   rocm.bindings.llvm.c.targetmachine.LLVMSetTargetMachineAsmVerbosity
   rocm.bindings.llvm.c.targetmachine.LLVMSetTargetMachineFastISel
   rocm.bindings.llvm.c.targetmachine.LLVMSetTargetMachineGlobalISel
   rocm.bindings.llvm.c.targetmachine.LLVMSetTargetMachineGlobalISelAbort
   rocm.bindings.llvm.c.targetmachine.LLVMSetTargetMachineMachineOutliner
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineEmitToFile
   rocm.bindings.llvm.c.targetmachine.LLVMTargetMachineEmitToMemoryBuffer
   rocm.bindings.llvm.c.targetmachine.LLVMGetDefaultTargetTriple
   rocm.bindings.llvm.c.targetmachine.LLVMNormalizeTargetTriple
   rocm.bindings.llvm.c.targetmachine.LLVMGetHostCPUName
   rocm.bindings.llvm.c.targetmachine.LLVMGetHostCPUFeatures
   rocm.bindings.llvm.c.targetmachine.LLVMAddAnalysisPasses


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpaqueTargetMachineOptions(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTargetMachineOptionsRef

.. py:class:: LLVMOpaqueTargetMachine(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTargetMachineRef

.. py:class:: LLVMTarget(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTargetRef

.. py:class:: LLVMCodeGenOptLevel

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMCodeGenLevelNone
      :type:  int


   .. py:attribute:: LLVMCodeGenLevelLess
      :type:  int


   .. py:attribute:: LLVMCodeGenLevelDefault
      :type:  int


   .. py:attribute:: LLVMCodeGenLevelAggressive
      :type:  int


.. py:class:: LLVMRelocMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMRelocDefault
      :type:  int


   .. py:attribute:: LLVMRelocStatic
      :type:  int


   .. py:attribute:: LLVMRelocPIC
      :type:  int


   .. py:attribute:: LLVMRelocDynamicNoPic
      :type:  int


   .. py:attribute:: LLVMRelocROPI
      :type:  int


   .. py:attribute:: LLVMRelocRWPI
      :type:  int


   .. py:attribute:: LLVMRelocROPI_RWPI
      :type:  int


.. py:class:: LLVMCodeModel

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMCodeModelDefault
      :type:  int


   .. py:attribute:: LLVMCodeModelJITDefault
      :type:  int


   .. py:attribute:: LLVMCodeModelTiny
      :type:  int


   .. py:attribute:: LLVMCodeModelSmall
      :type:  int


   .. py:attribute:: LLVMCodeModelKernel
      :type:  int


   .. py:attribute:: LLVMCodeModelMedium
      :type:  int


   .. py:attribute:: LLVMCodeModelLarge
      :type:  int


.. py:class:: LLVMCodeGenFileType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMAssemblyFile
      :type:  int


   .. py:attribute:: LLVMObjectFile
      :type:  int


.. py:class:: LLVMGlobalISelAbortMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMGlobalISelAbortEnable
      :type:  int


   .. py:attribute:: LLVMGlobalISelAbortDisable
      :type:  int


   .. py:attribute:: LLVMGlobalISelAbortDisableWithDiag
      :type:  int


.. py:function:: LLVMGetFirstTarget()

   Returns the first llvm::Target in the registered targets list.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetRef LLVMGetFirstTarget()


.. py:function:: LLVMGetNextTarget(T)

   Returns the next llvm::Target given a previous one (or null if there's none)

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetRef LLVMGetNextTarget(LLVMTargetRef T)


.. py:function:: LLVMGetTargetFromName(Name)

   Finds the target corresponding to the given name and stores it in ``T.``
   Returns 0 on success.

   Args:
       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetRef LLVMGetTargetFromName(const char * Name)


.. py:function:: LLVMGetTargetFromTriple(Triple)

   Finds the target corresponding to the given triple and stores it in ``T.``
   Returns 0 on success.

   Optionally returns any error in ErrorMessage.
   Use LLVMDisposeMessage to dispose the message.

   Args:
       Triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * T (:py:obj:`~.LLVMTarget`):
           (undocumented)
       * ErrorMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetTargetFromTriple(const char * Triple, LLVMTargetRef * T, char ** ErrorMessage)


.. py:function:: LLVMGetTargetName(T)

   Returns the name of a target. See llvm::Target::getName

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetTargetName(LLVMTargetRef T)


.. py:function:: LLVMGetTargetDescription(T)

   Returns the description  of a target. See llvm::Target::getDescription

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetTargetDescription(LLVMTargetRef T)


.. py:function:: LLVMTargetHasJIT(T)

   Returns if the target has a JIT

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTargetHasJIT(LLVMTargetRef T)


.. py:function:: LLVMTargetHasTargetMachine(T)

   Returns if the target has a TargetMachine associated

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTargetHasTargetMachine(LLVMTargetRef T)


.. py:function:: LLVMTargetHasAsmBackend(T)

   Returns if the target as an ASM backend (required for emitting output)

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTargetHasAsmBackend(LLVMTargetRef T)


.. py:function:: LLVMCreateTargetMachineOptions()

   Create a new set of options for an llvm::TargetMachine.

   The returned option structure must be released with
   LLVMDisposeTargetMachineOptions() after the call to
   LLVMCreateTargetMachineWithOptions().

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetMachineOptionsRef LLVMCreateTargetMachineOptions()


.. py:function:: LLVMDisposeTargetMachineOptions(Options)

   Dispose of an LLVMTargetMachineOptionsRef instance.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeTargetMachineOptions(LLVMTargetMachineOptionsRef Options)


.. py:function:: LLVMTargetMachineOptionsSetCPU(Options, CPU)

   Dispose of an LLVMTargetMachineOptionsRef instance.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       CPU (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetCPU(LLVMTargetMachineOptionsRef Options, const char * CPU)


.. py:function:: LLVMTargetMachineOptionsSetFeatures(Options, Features)

   Set the list of features for the target machine.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       Features (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           a comma-separated list of features.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetFeatures(LLVMTargetMachineOptionsRef Options, const char * Features)


.. py:function:: LLVMTargetMachineOptionsSetABI(Options, ABI)

   Set the list of features for the target machine.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       ABI (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetABI(LLVMTargetMachineOptionsRef Options, const char * ABI)


.. py:function:: LLVMTargetMachineOptionsSetCodeGenOptLevel(Options, Level)

   Set the list of features for the target machine.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       Level (:py:obj:`~.LLVMCodeGenOptLevel`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetCodeGenOptLevel(LLVMTargetMachineOptionsRef Options, LLVMCodeGenOptLevel Level)


.. py:function:: LLVMTargetMachineOptionsSetRelocMode(Options, Reloc)

   Set the list of features for the target machine.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       Reloc (:py:obj:`~.LLVMRelocMode`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetRelocMode(LLVMTargetMachineOptionsRef Options, LLVMRelocMode Reloc)


.. py:function:: LLVMTargetMachineOptionsSetCodeModel(Options, CodeModel)

   Set the list of features for the target machine.

   Args:
       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           (undocumented)

       CodeModel (:py:obj:`~.LLVMCodeModel`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMTargetMachineOptionsSetCodeModel(LLVMTargetMachineOptionsRef Options, LLVMCodeModel CodeModel)


.. py:function:: LLVMCreateTargetMachineWithOptions(T, Triple, Options)

   Create a new llvm::TargetMachine.

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           the target to create a machine for.

       Triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           a triple describing the target machine.

       Options (:py:obj:`~.LLVMOpaqueTargetMachineOptions`/:py:obj:`~.object`):
           additional configuration (see
           LLVMCreateTargetMachineOptions()).

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetMachineRef LLVMCreateTargetMachineWithOptions(LLVMTargetRef T, const char * Triple, LLVMTargetMachineOptionsRef Options)


.. py:function:: LLVMCreateTargetMachine(T, Triple, CPU, Features, Level, Reloc, CodeModel)

   Creates a new llvm::TargetMachine. See llvm::Target::createTargetMachine

   Args:
       T (:py:obj:`~.LLVMTarget`/:py:obj:`~.object`):
           (undocumented)

       Triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       CPU (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Features (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Level (:py:obj:`~.LLVMCodeGenOptLevel`):
           (undocumented)

       Reloc (:py:obj:`~.LLVMRelocMode`):
           (undocumented)

       CodeModel (:py:obj:`~.LLVMCodeModel`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetMachineRef LLVMCreateTargetMachine(LLVMTargetRef T, const char * Triple, const char * CPU, const char * Features, LLVMCodeGenOptLevel Level, LLVMRelocMode Reloc, LLVMCodeModel CodeModel)


.. py:function:: LLVMDisposeTargetMachine(T)

   Dispose the LLVMTargetMachineRef instance generated by
   LLVMCreateTargetMachine.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeTargetMachine(LLVMTargetMachineRef T)


.. py:function:: LLVMGetTargetMachineTarget(T)

   Returns the Target used in a TargetMachine

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetRef LLVMGetTargetMachineTarget(LLVMTargetMachineRef T)


.. py:function:: LLVMGetTargetMachineTriple(T)

   Returns the triple used creating this target machine.

   See
   llvm::TargetMachine::getTriple. The result needs to be disposed with
   LLVMDisposeMessage.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetTargetMachineTriple(LLVMTargetMachineRef T)


.. py:function:: LLVMGetTargetMachineCPU(T)

   Returns the cpu used creating this target machine.

   See
   llvm::TargetMachine::getCPU. The result needs to be disposed with
   LLVMDisposeMessage.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetTargetMachineCPU(LLVMTargetMachineRef T)


.. py:function:: LLVMGetTargetMachineFeatureString(T)

   Returns the feature string used creating this target machine.

   See
   llvm::TargetMachine::getFeatureString. The result needs to be disposed with
   LLVMDisposeMessage.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetTargetMachineFeatureString(LLVMTargetMachineRef T)


.. py:function:: LLVMCreateTargetDataLayout(T)

   Create a DataLayout based on the targetMachine.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetDataRef LLVMCreateTargetDataLayout(LLVMTargetMachineRef T)


.. py:function:: LLVMSetTargetMachineAsmVerbosity(T, VerboseAsm)

   Set the target machine's ASM verbosity.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       VerboseAsm (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTargetMachineAsmVerbosity(LLVMTargetMachineRef T, LLVMBool VerboseAsm)


.. py:function:: LLVMSetTargetMachineFastISel(T, Enable)

   Enable fast-path instruction selection.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       Enable (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTargetMachineFastISel(LLVMTargetMachineRef T, LLVMBool Enable)


.. py:function:: LLVMSetTargetMachineGlobalISel(T, Enable)

   Enable global instruction selection.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       Enable (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTargetMachineGlobalISel(LLVMTargetMachineRef T, LLVMBool Enable)


.. py:function:: LLVMSetTargetMachineGlobalISelAbort(T, Mode)

   Set abort behaviour when global instruction selection fails to lower/select
   an instruction.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       Mode (:py:obj:`~.LLVMGlobalISelAbortMode`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTargetMachineGlobalISelAbort(LLVMTargetMachineRef T, LLVMGlobalISelAbortMode Mode)


.. py:function:: LLVMSetTargetMachineMachineOutliner(T, Enable)

   Enable the MachineOutliner pass.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       Enable (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetTargetMachineMachineOutliner(LLVMTargetMachineRef T, LLVMBool Enable)


.. py:function:: LLVMTargetMachineEmitToFile(T, M, Filename, codegen, ErrorMessage)

   Emits an asm or object file for the given module to the filename.

   This
   wraps several c++ only classes (among them a file stream). Returns any
   error in ErrorMessage. Use LLVMDisposeMessage to dispose the message.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Filename (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       codegen (:py:obj:`~.LLVMCodeGenFileType`):
           (undocumented)

       ErrorMessage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTargetMachineEmitToFile(LLVMTargetMachineRef T, LLVMModuleRef M, const char * Filename, LLVMCodeGenFileType codegen, char ** ErrorMessage)


.. py:function:: LLVMTargetMachineEmitToMemoryBuffer(T, M, codegen, ErrorMessage, OutMemBuf)

   Compile the LLVM IR stored in ``M`` and store the result in ``OutMemBuf.``

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       codegen (:py:obj:`~.LLVMCodeGenFileType`):
           (undocumented)

       ErrorMessage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       OutMemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMTargetMachineEmitToMemoryBuffer(LLVMTargetMachineRef T, LLVMModuleRef M, LLVMCodeGenFileType codegen, char ** ErrorMessage, LLVMMemoryBufferRef * OutMemBuf)


.. py:function:: LLVMGetDefaultTargetTriple()

   Get a triple for the host machine as a string.

   The result needs to be
   disposed with LLVMDisposeMessage.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetDefaultTargetTriple()


.. py:function:: LLVMNormalizeTargetTriple(triple)

   Normalize a target triple.

   The result needs to be disposed with
   LLVMDisposeMessage.

   Args:
       triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMNormalizeTargetTriple(const char * triple)


.. py:function:: LLVMGetHostCPUName()

   Get the host CPU as a string.

   The result needs to be disposed with
   LLVMDisposeMessage.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetHostCPUName()


.. py:function:: LLVMGetHostCPUFeatures()

   Get the host CPU's features as a string.

   The result needs to be disposed
   with LLVMDisposeMessage.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetHostCPUFeatures()


.. py:function:: LLVMAddAnalysisPasses(T, PM)

   Adds the target-specific analysis passes to the pass manager.

   Args:
       T (:py:obj:`~.LLVMOpaqueTargetMachine`/:py:obj:`~.object`):
           (undocumented)

       PM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddAnalysisPasses(LLVMTargetMachineRef T, LLVMPassManagerRef PM)


