rocm.bindings.llvm.c.lljitutils
===============================

.. py:module:: rocm.bindings.llvm.c.lljitutils


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.lljitutils.has_symbol
   rocm.bindings.llvm.c.lljitutils.LLVMOrcLLJITEnableDebugSupport


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMOrcLLJITEnableDebugSupport(J)

   Install the plugin that submits debug objects to the executor.

   Executors must
   expose the llvm_orc_registerJITLoaderGDBAllocAction symbol.

   Args:
       J (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcLLJITEnableDebugSupport(LLVMOrcLLJITRef J)


