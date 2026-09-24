rocm.bindings.llvm.c.errorhandling
==================================

.. py:module:: rocm.bindings.llvm.c.errorhandling


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.errorhandling.LLVMFatalErrorHandler


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.errorhandling.has_symbol
   rocm.bindings.llvm.c.errorhandling.LLVMInstallFatalErrorHandler
   rocm.bindings.llvm.c.errorhandling.LLVMResetFatalErrorHandler
   rocm.bindings.llvm.c.errorhandling.LLVMEnablePrettyStackTrace


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMFatalErrorHandler(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: LLVMInstallFatalErrorHandler(Handler)

   Install a fatal error handler.

   By default, if LLVM detects a fatal error, it
   will call exit(1). This may not be appropriate in many contexts. For example,
   doing exit(1) will bypass many crash reporting/tracing system tools. This
   function allows you to install a callback that will be invoked prior to the
   call to exit(1).

   Args:
       Handler (:py:obj:`~.LLVMFatalErrorHandler`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInstallFatalErrorHandler(LLVMFatalErrorHandler Handler)


.. py:function:: LLVMResetFatalErrorHandler()

   Reset the fatal error handler.

   This resets LLVM's fatal error handling
   behavior to the default.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMResetFatalErrorHandler()


.. py:function:: LLVMEnablePrettyStackTrace()

   Enable LLVM's built-in stack trace code.

   This intercepts the OS's crash
   signals and prints which component of LLVM you were in at the time if the
   crash.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMEnablePrettyStackTrace()


