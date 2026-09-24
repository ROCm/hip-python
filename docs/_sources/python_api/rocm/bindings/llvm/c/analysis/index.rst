rocm.bindings.llvm.c.analysis
=============================

.. py:module:: rocm.bindings.llvm.c.analysis


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.analysis.LLVMVerifierFailureAction


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.analysis.has_symbol
   rocm.bindings.llvm.c.analysis.LLVMVerifyModule
   rocm.bindings.llvm.c.analysis.LLVMVerifyFunction
   rocm.bindings.llvm.c.analysis.LLVMViewFunctionCFG
   rocm.bindings.llvm.c.analysis.LLVMViewFunctionCFGOnly


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMVerifierFailureAction

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMAbortProcessAction
      :type:  int


   .. py:attribute:: LLVMPrintMessageAction
      :type:  int


   .. py:attribute:: LLVMReturnStatusAction
      :type:  int


.. py:function:: LLVMVerifyModule(M, Action)

   Verifies that a module is valid, taking the specified action if not.

   Optionally returns a human-readable description of any invalid constructs.
   OutMessage must be disposed with LLVMDisposeMessage.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Action (:py:obj:`~.LLVMVerifierFailureAction`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.int`: (undocumented)
       * OutMessage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMVerifyModule(LLVMModuleRef M, LLVMVerifierFailureAction Action, char ** OutMessage)


.. py:function:: LLVMVerifyFunction(Fn, Action)

   Verifies that a single function is valid, taking the specified action.

   Useful
   for debugging.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Action (:py:obj:`~.LLVMVerifierFailureAction`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMVerifyFunction(LLVMValueRef Fn, LLVMVerifierFailureAction Action)


.. py:function:: LLVMViewFunctionCFG(Fn)

   Open up a ghostview window that displays the CFG of the current function.

   Useful for debugging.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMViewFunctionCFG(LLVMValueRef Fn)


.. py:function:: LLVMViewFunctionCFGOnly(Fn)

   Open up a ghostview window that displays the CFG of the current function.

   Useful for debugging.

   Args:
       Fn (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMViewFunctionCFGOnly(LLVMValueRef Fn)


