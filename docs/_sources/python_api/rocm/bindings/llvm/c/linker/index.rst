rocm.bindings.llvm.c.linker
===========================

.. py:module:: rocm.bindings.llvm.c.linker


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.linker.LLVMLinkerMode


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.linker.has_symbol
   rocm.bindings.llvm.c.linker.LLVMLinkModules2


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMLinkerMode

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMLinkerDestroySource
      :type:  int


   .. py:attribute:: LLVMLinkerPreserveSource_Removed
      :type:  int


.. py:function:: LLVMLinkModules2(Dest, Src)

   Links the source module into the destination module.

   The source module is
   destroyed.
   The return value is true if an error occurred, false otherwise.
   Use the diagnostic handler to get any diagnostic message.

   Args:
       Dest (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Src (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMLinkModules2(LLVMModuleRef Dest, LLVMModuleRef Src)


