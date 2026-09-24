rocm.bindings.llvm.c.bitwriter
==============================

.. py:module:: rocm.bindings.llvm.c.bitwriter


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.bitwriter.has_symbol
   rocm.bindings.llvm.c.bitwriter.LLVMWriteBitcodeToFile
   rocm.bindings.llvm.c.bitwriter.LLVMWriteBitcodeToFD
   rocm.bindings.llvm.c.bitwriter.LLVMWriteBitcodeToFileHandle
   rocm.bindings.llvm.c.bitwriter.LLVMWriteBitcodeToMemoryBuffer


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMWriteBitcodeToFile(M, Path)

   Writes a module to the specified path. Returns 0 on success.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char * Path)


.. py:function:: LLVMWriteBitcodeToFD(M, FD, ShouldClose, Unbuffered)

   Writes a module to an open file descriptor. Returns 0 on success.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       FD (:py:obj:`~.int`):
           (undocumented)

       ShouldClose (:py:obj:`~.int`):
           (undocumented)

       Unbuffered (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMWriteBitcodeToFD(LLVMModuleRef M, int FD, int ShouldClose, int Unbuffered)


.. py:function:: LLVMWriteBitcodeToFileHandle(M, Handle)

   Deprecated for LLVMWriteBitcodeToFD.

   Writes a module to an open file
   descriptor. Returns 0 on success. Closes the Handle.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Handle (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int Handle)


.. py:function:: LLVMWriteBitcodeToMemoryBuffer(M)

   Writes a module to a new memory buffer and returns it.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMemoryBufferRef LLVMWriteBitcodeToMemoryBuffer(LLVMModuleRef M)


