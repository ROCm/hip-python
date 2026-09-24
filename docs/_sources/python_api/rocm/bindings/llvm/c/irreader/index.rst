rocm.bindings.llvm.c.irreader
=============================

.. py:module:: rocm.bindings.llvm.c.irreader


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.irreader.has_symbol
   rocm.bindings.llvm.c.irreader.LLVMParseIRInContext
   rocm.bindings.llvm.c.irreader.LLVMParseIRInContext2


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMParseIRInContext(ContextRef, MemBuf)

   Read LLVM IR from a memory buffer and convert it into an in-memory Module
   object.

   Returns 0 on success.
   Optionally returns a human-readable description of any errors that
   occurred during parsing IR. OutMessage must be disposed with
   LLVMDisposeMessage.
   The memory buffer is consumed by this function.
   This is deprecated. Use LLVMParseIRInContext2 instead.

   See:
       llvm::ParseIR()

   Args:
       ContextRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutM (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseIRInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM, char ** OutMessage)


.. py:function:: LLVMParseIRInContext2(ContextRef, MemBuf)

   Read LLVM IR from a memory buffer and convert it into an in-memory Module
   object.

   Returns 0 on success.
   Optionally returns a human-readable description of any errors that
   occurred during parsing IR. OutMessage must be disposed with
   LLVMDisposeMessage.
   The memory buffer is not consumed by this function. It is the responsibility
   of the caller to free it with ``LLVMDisposeMemoryBuffer.``

   See:
       llvm::ParseIR()

   Args:
       ContextRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutM (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseIRInContext2(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM, char ** OutMessage)


