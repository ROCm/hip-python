rocm.bindings.llvm.c.bitreader
==============================

.. py:module:: rocm.bindings.llvm.c.bitreader


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.bitreader.has_symbol
   rocm.bindings.llvm.c.bitreader.LLVMParseBitcode
   rocm.bindings.llvm.c.bitreader.LLVMParseBitcode2
   rocm.bindings.llvm.c.bitreader.LLVMParseBitcodeInContext
   rocm.bindings.llvm.c.bitreader.LLVMParseBitcodeInContext2
   rocm.bindings.llvm.c.bitreader.LLVMGetBitcodeModuleInContext
   rocm.bindings.llvm.c.bitreader.LLVMGetBitcodeModuleInContext2
   rocm.bindings.llvm.c.bitreader.LLVMGetBitcodeModule
   rocm.bindings.llvm.c.bitreader.LLVMGetBitcodeModule2


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMParseBitcode(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutModule (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseBitcode(LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutModule, char ** OutMessage)


.. py:function:: LLVMParseBitcode2(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutModule (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseBitcode2(LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutModule)


.. py:function:: LLVMParseBitcodeInContext(ContextRef, MemBuf)

   This is deprecated. Use LLVMParseBitcodeInContext2.

   Args:
       ContextRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutModule (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseBitcodeInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutModule, char ** OutMessage)


.. py:function:: LLVMParseBitcodeInContext2(ContextRef, MemBuf)

   This is deprecated. Use LLVMParseBitcodeInContext2.

   Args:
       ContextRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutModule (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMParseBitcodeInContext2(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutModule)


.. py:function:: LLVMGetBitcodeModuleInContext(ContextRef, MemBuf)

   Reads a module from the specified path, returning via the OutMP parameter
   a module provider which performs lazy deserialization.

   Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage.
   This is deprecated. Use LLVMGetBitcodeModuleInContext2.

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

       LLVMBool LLVMGetBitcodeModuleInContext(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM, char ** OutMessage)


.. py:function:: LLVMGetBitcodeModuleInContext2(ContextRef, MemBuf)

   Reads a module from the given memory buffer, returning via the OutMP
   parameter a module provider which performs lazy deserialization.

   Returns 0 on success.

   Takes ownership of ``MemBuf`` if (and only if) the module was read
   successfully.

   Args:
       ContextRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutM (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetBitcodeModuleInContext2(LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM)


.. py:function:: LLVMGetBitcodeModule(MemBuf)

   (No short description, might be part of a group.)

   Args:
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

       LLVMBool LLVMGetBitcodeModule(LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM, char ** OutMessage)


.. py:function:: LLVMGetBitcodeModule2(MemBuf)

   (No short description, might be part of a group.)

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutM (:py:obj:`~.LLVMOpaqueModule`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetBitcodeModule2(LLVMMemoryBufferRef MemBuf, LLVMModuleRef * OutM)


