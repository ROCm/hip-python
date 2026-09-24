rocm.bindings.llvm.c.error
==========================

.. py:module:: rocm.bindings.llvm.c.error


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.error.LLVMErrorRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.error.LLVMOpaqueError


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.error.has_symbol
   rocm.bindings.llvm.c.error.LLVMGetErrorTypeId
   rocm.bindings.llvm.c.error.LLVMConsumeError
   rocm.bindings.llvm.c.error.LLVMCantFail
   rocm.bindings.llvm.c.error.LLVMGetErrorMessage
   rocm.bindings.llvm.c.error.LLVMDisposeErrorMessage
   rocm.bindings.llvm.c.error.LLVMGetStringErrorTypeId
   rocm.bindings.llvm.c.error.LLVMCreateStringError


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpaqueError(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMErrorRef

.. py:function:: LLVMGetErrorTypeId(Err)

   Returns the type id for the given error instance, which must be a failure
   value (i.e.

   non-null).

   Args:
       Err (:py:obj:`~.LLVMOpaqueError`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorTypeId LLVMGetErrorTypeId(LLVMErrorRef Err)


.. py:function:: LLVMConsumeError(Err)

   Dispose of the given error without handling it.

   This operation consumes the
   error, and the given LLVMErrorRef value is not usable once this call returns.
   Note: This method *only* needs to be called if the error is not being passed
   to some other consuming operation, e.g. LLVMGetErrorMessage.

   Args:
       Err (:py:obj:`~.LLVMOpaqueError`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMConsumeError(LLVMErrorRef Err)


.. py:function:: LLVMCantFail(Err)

   Report a fatal error if Err is a failure value.

   This function can be used to wrap calls to fallible functions ONLY when it is
   known that the Error will always be a success value.

   Args:
       Err (:py:obj:`~.LLVMOpaqueError`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMCantFail(LLVMErrorRef Err)


.. py:function:: LLVMGetErrorMessage(Err)

   Returns the given string's error message.

   This operation consumes the error,
   and the given LLVMErrorRef value is not usable once this call returns.
   The caller is responsible for disposing of the string by calling
   LLVMDisposeErrorMessage.

   Args:
       Err (:py:obj:`~.LLVMOpaqueError`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMGetErrorMessage(LLVMErrorRef Err)


.. py:function:: LLVMDisposeErrorMessage(ErrMsg)

   Dispose of the given error message.

   Args:
       ErrMsg (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeErrorMessage(char * ErrMsg)


.. py:function:: LLVMGetStringErrorTypeId()

   Returns the type id for llvm StringError.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorTypeId LLVMGetStringErrorTypeId()


.. py:function:: LLVMCreateStringError(ErrMsg)

   Create a StringError.

   Args:
       ErrMsg (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMCreateStringError(const char * ErrMsg)


