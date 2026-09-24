rocm.bindings.llvm.c.support
============================

.. py:module:: rocm.bindings.llvm.c.support


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.support.has_symbol
   rocm.bindings.llvm.c.support.LLVMLoadLibraryPermanently
   rocm.bindings.llvm.c.support.LLVMParseCommandLineOptions
   rocm.bindings.llvm.c.support.LLVMSearchForAddressOfSymbol
   rocm.bindings.llvm.c.support.LLVMAddSymbol


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:function:: LLVMLoadLibraryPermanently(Filename)

   This function permanently loads the dynamic library at the given path.

   It is safe to call this function multiple times for the same library.

   See:
       sys::DynamicLibrary::LoadLibraryPermanently()

   Args:
       Filename (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMLoadLibraryPermanently(const char * Filename)


.. py:function:: LLVMParseCommandLineOptions(argc, argv, Overview)

   This function parses the given arguments using the LLVM command line parser.

   Note that the only stable thing about this function is its signature; you
   cannot rely on any particular set of command line arguments being interpreted
   the same way across LLVM versions.

   See:
       llvm::cl::ParseCommandLineOptions()

   Args:
       argc (:py:obj:`~.int`):
           (undocumented)

       argv (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Overview (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMParseCommandLineOptions(int argc, const char *const * argv, const char * Overview)


.. py:function:: LLVMSearchForAddressOfSymbol(symbolName)

   This function will search through all previously loaded dynamic
   libraries for the symbol ``symbolName.`` If it is found, the address of
   that symbol is returned.

   If not, null is returned.

   See:
       sys::DynamicLibrary::SearchForAddressOfSymbol()

   Args:
       symbolName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void * LLVMSearchForAddressOfSymbol(const char * symbolName)


.. py:function:: LLVMAddSymbol(symbolName, symbolValue)

   This functions permanently adds the symbol ``symbolName`` with the
   value ``symbolValue.``  These symbols are searched before any
   libraries.

   See:
       sys::DynamicLibrary::AddSymbol()

   Args:
       symbolName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       symbolValue (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMAddSymbol(const char * symbolName, void * symbolValue)


