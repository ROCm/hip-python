rocm.bindings.llvm.c.comdat
===========================

.. py:module:: rocm.bindings.llvm.c.comdat


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.comdat.LLVMComdatSelectionKind


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.comdat.has_symbol
   rocm.bindings.llvm.c.comdat.LLVMGetOrInsertComdat
   rocm.bindings.llvm.c.comdat.LLVMGetComdat
   rocm.bindings.llvm.c.comdat.LLVMSetComdat
   rocm.bindings.llvm.c.comdat.LLVMGetComdatSelectionKind
   rocm.bindings.llvm.c.comdat.LLVMSetComdatSelectionKind


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMComdatSelectionKind

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMAnyComdatSelectionKind
      :type:  int


   .. py:attribute:: LLVMExactMatchComdatSelectionKind
      :type:  int


   .. py:attribute:: LLVMLargestComdatSelectionKind
      :type:  int


   .. py:attribute:: LLVMNoDeduplicateComdatSelectionKind
      :type:  int


   .. py:attribute:: LLVMSameSizeComdatSelectionKind
      :type:  int


.. py:function:: LLVMGetOrInsertComdat(M, Name)

   Return the Comdat in the module with the specified name.

   It is created
   if it didn't already exist.

   See:
       llvm::Module::getOrInsertComdat()

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMComdatRef LLVMGetOrInsertComdat(LLVMModuleRef M, const char * Name)


.. py:function:: LLVMGetComdat(V)

   Get the Comdat assigned to the given global object.

   See:
       llvm::GlobalObject::getComdat()

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMComdatRef LLVMGetComdat(LLVMValueRef V)


.. py:function:: LLVMSetComdat(V, C)

   Assign the Comdat to the given global object.

   See:
       llvm::GlobalObject::setComdat()

   Args:
       V (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetComdat(LLVMValueRef V, LLVMComdatRef C)


.. py:function:: LLVMGetComdatSelectionKind(C)

   Get the conflict resolution selection kind for the Comdat.

   See:
       llvm::Comdat::getSelectionKind()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMComdatSelectionKind`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMComdatSelectionKind LLVMGetComdatSelectionKind(LLVMComdatRef C)


.. py:function:: LLVMSetComdatSelectionKind(C, Kind)

   Set the conflict resolution selection kind for the Comdat.

   See:
       llvm::Comdat::setSelectionKind()

   Args:
       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Kind (:py:obj:`~.LLVMComdatSelectionKind`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetComdatSelectionKind(LLVMComdatRef C, LLVMComdatSelectionKind Kind)


