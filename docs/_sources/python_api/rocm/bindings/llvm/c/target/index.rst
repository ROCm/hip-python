rocm.bindings.llvm.c.target
===========================

.. py:module:: rocm.bindings.llvm.c.target


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.target.LLVMTargetDataRef
   rocm.bindings.llvm.c.target.LLVMTargetLibraryInfoRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.target.LLVMByteOrdering
   rocm.bindings.llvm.c.target.LLVMOpaqueTargetData
   rocm.bindings.llvm.c.target.LLVMOpaqueTargetLibraryInfotData


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.target.has_symbol
   rocm.bindings.llvm.c.target.LLVMInitializeAllTargetInfos
   rocm.bindings.llvm.c.target.LLVMInitializeAllTargets
   rocm.bindings.llvm.c.target.LLVMInitializeAllTargetMCs
   rocm.bindings.llvm.c.target.LLVMInitializeAllAsmPrinters
   rocm.bindings.llvm.c.target.LLVMInitializeAllAsmParsers
   rocm.bindings.llvm.c.target.LLVMInitializeAllDisassemblers
   rocm.bindings.llvm.c.target.LLVMInitializeNativeTarget
   rocm.bindings.llvm.c.target.LLVMInitializeNativeAsmParser
   rocm.bindings.llvm.c.target.LLVMInitializeNativeAsmPrinter
   rocm.bindings.llvm.c.target.LLVMInitializeNativeDisassembler
   rocm.bindings.llvm.c.target.LLVMGetModuleDataLayout
   rocm.bindings.llvm.c.target.LLVMSetModuleDataLayout
   rocm.bindings.llvm.c.target.LLVMCreateTargetData
   rocm.bindings.llvm.c.target.LLVMDisposeTargetData
   rocm.bindings.llvm.c.target.LLVMAddTargetLibraryInfo
   rocm.bindings.llvm.c.target.LLVMCopyStringRepOfTargetData
   rocm.bindings.llvm.c.target.LLVMByteOrder
   rocm.bindings.llvm.c.target.LLVMPointerSize
   rocm.bindings.llvm.c.target.LLVMPointerSizeForAS
   rocm.bindings.llvm.c.target.LLVMIntPtrType
   rocm.bindings.llvm.c.target.LLVMIntPtrTypeForAS
   rocm.bindings.llvm.c.target.LLVMIntPtrTypeInContext
   rocm.bindings.llvm.c.target.LLVMIntPtrTypeForASInContext
   rocm.bindings.llvm.c.target.LLVMSizeOfTypeInBits
   rocm.bindings.llvm.c.target.LLVMStoreSizeOfType
   rocm.bindings.llvm.c.target.LLVMABISizeOfType
   rocm.bindings.llvm.c.target.LLVMABIAlignmentOfType
   rocm.bindings.llvm.c.target.LLVMCallFrameAlignmentOfType
   rocm.bindings.llvm.c.target.LLVMPreferredAlignmentOfType
   rocm.bindings.llvm.c.target.LLVMPreferredAlignmentOfGlobal
   rocm.bindings.llvm.c.target.LLVMElementAtOffset
   rocm.bindings.llvm.c.target.LLVMOffsetOfElement


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMByteOrdering

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMBigEndian
      :type:  int


   .. py:attribute:: LLVMLittleEndian
      :type:  int


.. py:class:: LLVMOpaqueTargetData(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTargetDataRef

.. py:class:: LLVMOpaqueTargetLibraryInfotData(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMTargetLibraryInfoRef

.. py:function:: LLVMInitializeAllTargetInfos()

   LLVMInitializeAllTargetInfos - The main program should call this function if
   it wants access to all available targets that LLVM is configured to
   support.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllTargetInfos()


.. py:function:: LLVMInitializeAllTargets()

   LLVMInitializeAllTargets - The main program should call this function if it
   wants to link in all available targets that LLVM is configured to
   support.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllTargets()


.. py:function:: LLVMInitializeAllTargetMCs()

   LLVMInitializeAllTargetMCs - The main program should call this function if
   it wants access to all available target MC that LLVM is configured to
   support.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllTargetMCs()


.. py:function:: LLVMInitializeAllAsmPrinters()

   LLVMInitializeAllAsmPrinters - The main program should call this function if
   it wants all asm printers that LLVM is configured to support, to make them
   available via the TargetRegistry.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllAsmPrinters()


.. py:function:: LLVMInitializeAllAsmParsers()

   LLVMInitializeAllAsmParsers - The main program should call this function if
   it wants all asm parsers that LLVM is configured to support, to make them
   available via the TargetRegistry.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllAsmParsers()


.. py:function:: LLVMInitializeAllDisassemblers()

   LLVMInitializeAllDisassemblers - The main program should call this function
   if it wants all disassemblers that LLVM is configured to support, to make
   them available via the TargetRegistry.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInitializeAllDisassemblers()


.. py:function:: LLVMInitializeNativeTarget()

   LLVMInitializeNativeTarget - The main program should call this function to
   initialize the native target corresponding to the host.

   This is useful
   for JIT applications to ensure that the target gets linked in correctly.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMInitializeNativeTarget()


.. py:function:: LLVMInitializeNativeAsmParser()

   LLVMInitializeNativeTargetAsmParser - The main program should call this
   function to initialize the parser for the native target corresponding to the
   host.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMInitializeNativeAsmParser()


.. py:function:: LLVMInitializeNativeAsmPrinter()

   LLVMInitializeNativeTargetAsmPrinter - The main program should call this
   function to initialize the printer for the native target corresponding to
   the host.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMInitializeNativeAsmPrinter()


.. py:function:: LLVMInitializeNativeDisassembler()

   LLVMInitializeNativeTargetDisassembler - The main program should call this
   function to initialize the disassembler for the native target corresponding
   to the host.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMInitializeNativeDisassembler()


.. py:function:: LLVMGetModuleDataLayout(M)

   Obtain the data layout for a module.

   See:
       Module::getDataLayout()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetDataRef LLVMGetModuleDataLayout(LLVMModuleRef M)


.. py:function:: LLVMSetModuleDataLayout(M, DL)

   Set the data layout for a module.

   See:
       Module::setDataLayout()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       DL (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetModuleDataLayout(LLVMModuleRef M, LLVMTargetDataRef DL)


.. py:function:: LLVMCreateTargetData(StringRep)

   Creates target data from a target layout string.

   See the constructor llvm::DataLayout::DataLayout.

   Args:
       StringRep (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTargetDataRef LLVMCreateTargetData(const char * StringRep)


.. py:function:: LLVMDisposeTargetData(TD)

   Deallocates a TargetData.

   See the destructor llvm::DataLayout::~DataLayout.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeTargetData(LLVMTargetDataRef TD)


.. py:function:: LLVMAddTargetLibraryInfo(TLI, PM)

   Adds target library information to a pass manager.

   This does not take
   ownership of the target library info.
   See the method llvm::PassManagerBase::add.

   Args:
       TLI (:py:obj:`~.LLVMOpaqueTargetLibraryInfotData`/:py:obj:`~.object`):
           (undocumented)

       PM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddTargetLibraryInfo(LLVMTargetLibraryInfoRef TLI, LLVMPassManagerRef PM)


.. py:function:: LLVMCopyStringRepOfTargetData(TD)

   Converts target data to a target layout string.

   The string must be disposed
   with LLVMDisposeMessage.
   See the constructor llvm::DataLayout::DataLayout.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMCopyStringRepOfTargetData(LLVMTargetDataRef TD)


.. py:function:: LLVMByteOrder(TD)

   Returns the byte order of a target, either LLVMBigEndian or
   LLVMLittleEndian.

   See the method llvm::DataLayout::isLittleEndian.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMByteOrdering`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       enum LLVMByteOrdering LLVMByteOrder(LLVMTargetDataRef TD)


.. py:function:: LLVMPointerSize(TD)

   Returns the pointer size in bytes for a target.

   See the method llvm::DataLayout::getPointerSize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMPointerSize(LLVMTargetDataRef TD)


.. py:function:: LLVMPointerSizeForAS(TD, AS)

   Returns the pointer size in bytes for a target for a specified
   address space.

   See the method llvm::DataLayout::getPointerSize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       AS (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMPointerSizeForAS(LLVMTargetDataRef TD, unsigned int AS)


.. py:function:: LLVMIntPtrType(TD)

   Returns the integer type that is the same size as a pointer on a target.

   See the method llvm::DataLayout::getIntPtrType.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntPtrType(LLVMTargetDataRef TD)


.. py:function:: LLVMIntPtrTypeForAS(TD, AS)

   Returns the integer type that is the same size as a pointer on a target.

   This version allows the address space to be specified.
   See the method llvm::DataLayout::getIntPtrType.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       AS (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntPtrTypeForAS(LLVMTargetDataRef TD, unsigned int AS)


.. py:function:: LLVMIntPtrTypeInContext(C, TD)

   Returns the integer type that is the same size as a pointer on a target.

   See the method llvm::DataLayout::getIntPtrType.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntPtrTypeInContext(LLVMContextRef C, LLVMTargetDataRef TD)


.. py:function:: LLVMIntPtrTypeForASInContext(C, TD, AS)

   Returns the integer type that is the same size as a pointer on a target.

   This version allows the address space to be specified.
   See the method llvm::DataLayout::getIntPtrType.

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       AS (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMTypeRef LLVMIntPtrTypeForASInContext(LLVMContextRef C, LLVMTargetDataRef TD, unsigned int AS)


.. py:function:: LLVMSizeOfTypeInBits(TD, Ty)

   Computes the size of a type in bits for a target.

   See the method llvm::DataLayout::getTypeSizeInBits.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMSizeOfTypeInBits(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMStoreSizeOfType(TD, Ty)

   Computes the storage size of a type in bytes for a target.

   See the method llvm::DataLayout::getTypeStoreSize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMStoreSizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMABISizeOfType(TD, Ty)

   Computes the ABI size of a type in bytes for a target.

   See the method llvm::DataLayout::getTypeAllocSize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMABISizeOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMABIAlignmentOfType(TD, Ty)

   Computes the ABI alignment of a type in bytes for a target.

   See the method llvm::DataLayout::getTypeABISize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMABIAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMCallFrameAlignmentOfType(TD, Ty)

   Computes the call frame alignment of a type in bytes for a target.

   See the method llvm::DataLayout::getTypeABISize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMCallFrameAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMPreferredAlignmentOfType(TD, Ty)

   Computes the preferred alignment of a type in bytes for a target.

   See the method llvm::DataLayout::getTypeABISize.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMPreferredAlignmentOfType(LLVMTargetDataRef TD, LLVMTypeRef Ty)


.. py:function:: LLVMPreferredAlignmentOfGlobal(TD, GlobalVar)

   Computes the preferred alignment of a global variable in bytes for a target.

   See the method llvm::DataLayout::getPreferredAlignment.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       GlobalVar (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMPreferredAlignmentOfGlobal(LLVMTargetDataRef TD, LLVMValueRef GlobalVar)


.. py:function:: LLVMElementAtOffset(TD, StructTy, Offset)

   Computes the structure element that contains the byte offset for a target.

   See the method llvm::StructLayout::getElementContainingOffset.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Offset (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMElementAtOffset(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned long long Offset)


.. py:function:: LLVMOffsetOfElement(TD, StructTy, Element)

   Computes the byte offset of the indexed struct element for a target.

   See the method llvm::StructLayout::getElementContainingOffset.

   Args:
       TD (:py:obj:`~.LLVMOpaqueTargetData`/:py:obj:`~.object`):
           (undocumented)

       StructTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Element (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned long long LLVMOffsetOfElement(LLVMTargetDataRef TD, LLVMTypeRef StructTy, unsigned int Element)


