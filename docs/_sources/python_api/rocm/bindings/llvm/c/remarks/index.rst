rocm.bindings.llvm.c.remarks
============================

.. py:module:: rocm.bindings.llvm.c.remarks


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.remarks.LLVMRemarkStringRef
   rocm.bindings.llvm.c.remarks.LLVMRemarkDebugLocRef
   rocm.bindings.llvm.c.remarks.LLVMRemarkArgRef
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryRef
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.remarks.LLVMRemarkType
   rocm.bindings.llvm.c.remarks.LLVMRemarkOpaqueString
   rocm.bindings.llvm.c.remarks.LLVMRemarkOpaqueDebugLoc
   rocm.bindings.llvm.c.remarks.LLVMRemarkOpaqueArg
   rocm.bindings.llvm.c.remarks.LLVMRemarkOpaqueEntry
   rocm.bindings.llvm.c.remarks.LLVMRemarkOpaqueParser


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.remarks.has_symbol
   rocm.bindings.llvm.c.remarks.LLVMRemarkStringGetData
   rocm.bindings.llvm.c.remarks.LLVMRemarkStringGetLen
   rocm.bindings.llvm.c.remarks.LLVMRemarkDebugLocGetSourceFilePath
   rocm.bindings.llvm.c.remarks.LLVMRemarkDebugLocGetSourceLine
   rocm.bindings.llvm.c.remarks.LLVMRemarkDebugLocGetSourceColumn
   rocm.bindings.llvm.c.remarks.LLVMRemarkArgGetKey
   rocm.bindings.llvm.c.remarks.LLVMRemarkArgGetValue
   rocm.bindings.llvm.c.remarks.LLVMRemarkArgGetDebugLoc
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryDispose
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetType
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetPassName
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetRemarkName
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetFunctionName
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetDebugLoc
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetHotness
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetNumArgs
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetFirstArg
   rocm.bindings.llvm.c.remarks.LLVMRemarkEntryGetNextArg
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserCreateYAML
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserCreateBitstream
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserGetNext
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserHasError
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserGetErrorMessage
   rocm.bindings.llvm.c.remarks.LLVMRemarkParserDispose
   rocm.bindings.llvm.c.remarks.LLVMRemarkVersion


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMRemarkType

   Bases: :py:obj:`enum.IntEnum`


   The type of the emitted remark.
       


   .. py:attribute:: LLVMRemarkTypeUnknown
      :type:  int


   .. py:attribute:: LLVMRemarkTypePassed
      :type:  int


   .. py:attribute:: LLVMRemarkTypeMissed
      :type:  int


   .. py:attribute:: LLVMRemarkTypeAnalysis
      :type:  int


   .. py:attribute:: LLVMRemarkTypeAnalysisFPCommute
      :type:  int


   .. py:attribute:: LLVMRemarkTypeAnalysisAliasing
      :type:  int


   .. py:attribute:: LLVMRemarkTypeFailure
      :type:  int


.. py:class:: LLVMRemarkOpaqueString(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRemarkStringRef

.. py:function:: LLVMRemarkStringGetData(String)

   Returns the buffer holding the string.

   Since:
       REMARKS_API_VERSION=0

   Args:
       String (:py:obj:`~.LLVMRemarkOpaqueString`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMRemarkStringGetData(LLVMRemarkStringRef String)


.. py:function:: LLVMRemarkStringGetLen(String)

   Returns the size of the string.

   Since:
       REMARKS_API_VERSION=0

   Args:
       String (:py:obj:`~.LLVMRemarkOpaqueString`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMRemarkStringGetLen(LLVMRemarkStringRef String)


.. py:class:: LLVMRemarkOpaqueDebugLoc(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRemarkDebugLocRef

.. py:function:: LLVMRemarkDebugLocGetSourceFilePath(DL)

   Return the path to the source file for a debug location.

   Since:
       REMARKS_API_VERSION=0

   Args:
       DL (:py:obj:`~.LLVMRemarkOpaqueDebugLoc`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkDebugLocGetSourceFilePath(LLVMRemarkDebugLocRef DL)


.. py:function:: LLVMRemarkDebugLocGetSourceLine(DL)

   Return the line in the source file for a debug location.

   Since:
       REMARKS_API_VERSION=0

   Args:
       DL (:py:obj:`~.LLVMRemarkOpaqueDebugLoc`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMRemarkDebugLocGetSourceLine(LLVMRemarkDebugLocRef DL)


.. py:function:: LLVMRemarkDebugLocGetSourceColumn(DL)

   Return the column in the source file for a debug location.

   Since:
       REMARKS_API_VERSION=0

   Args:
       DL (:py:obj:`~.LLVMRemarkOpaqueDebugLoc`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMRemarkDebugLocGetSourceColumn(LLVMRemarkDebugLocRef DL)


.. py:class:: LLVMRemarkOpaqueArg(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRemarkArgRef

.. py:function:: LLVMRemarkArgGetKey(Arg)

   Returns the key of an argument.

   The key defines what the value is, and the
   same key can appear multiple times in the list of arguments.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Arg (:py:obj:`~.LLVMRemarkOpaqueArg`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkArgGetKey(LLVMRemarkArgRef Arg)


.. py:function:: LLVMRemarkArgGetValue(Arg)

   Returns the value of an argument. This is a string that can contain newlines.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Arg (:py:obj:`~.LLVMRemarkOpaqueArg`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkArgGetValue(LLVMRemarkArgRef Arg)


.. py:function:: LLVMRemarkArgGetDebugLoc(Arg)

   Returns the debug location that is attached to the value of this argument.

   If there is no debug location, the return value will be `NULL`.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Arg (:py:obj:`~.LLVMRemarkOpaqueArg`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkDebugLocRef LLVMRemarkArgGetDebugLoc(LLVMRemarkArgRef Arg)


.. py:class:: LLVMRemarkOpaqueEntry(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRemarkEntryRef

.. py:function:: LLVMRemarkEntryDispose(Remark)

   Free the resources used by the remark entry.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemarkEntryDispose(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetType(Remark)

   The type of the remark.

   For example, it can allow users to only keep the
   missed optimizations from the compiler.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMRemarkType`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       enum LLVMRemarkType LLVMRemarkEntryGetType(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetPassName(Remark)

   Get the name of the pass that emitted this remark.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkEntryGetPassName(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetRemarkName(Remark)

   Get an identifier of the remark.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkEntryGetRemarkName(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetFunctionName(Remark)

   Get the name of the function being processed when the remark was emitted.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkStringRef LLVMRemarkEntryGetFunctionName(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetDebugLoc(Remark)

   Returns the debug location that is attached to this remark.

   If there is no debug location, the return value will be `NULL`.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkDebugLocRef LLVMRemarkEntryGetDebugLoc(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetHotness(Remark)

   Return the hotness of the remark.

   A hotness of `0` means this value is not set.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMRemarkEntryGetHotness(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetNumArgs(Remark)

   The number of arguments the remark holds.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMRemarkEntryGetNumArgs(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetFirstArg(Remark)

   Get a new iterator to iterate over a remark's argument.

   If there are no arguments in ``Remark,`` the return value will be `NULL`.

   The lifetime of the returned value is bound to the lifetime of ``Remark.``

   Since:
       REMARKS_API_VERSION=0

   Args:
       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkArgRef LLVMRemarkEntryGetFirstArg(LLVMRemarkEntryRef Remark)


.. py:function:: LLVMRemarkEntryGetNextArg(It, Remark)

   Get the next argument in ``Remark`` from the position of ``It.``

   Returns `NULL` if there are no more arguments available.

   The lifetime of the returned value is bound to the lifetime of ``Remark.``

   Since:
       REMARKS_API_VERSION=0

   Args:
       It (:py:obj:`~.LLVMRemarkOpaqueArg`/:py:obj:`~.object`):
           (undocumented)

       Remark (:py:obj:`~.LLVMRemarkOpaqueEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkArgRef LLVMRemarkEntryGetNextArg(LLVMRemarkArgRef It, LLVMRemarkEntryRef Remark)


.. py:class:: LLVMRemarkOpaqueParser(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRemarkParserRef

.. py:function:: LLVMRemarkParserCreateYAML(Buf, Size)

   Creates a remark parser that can be used to parse the buffer located in \p
   Buf of size ``Size`` bytes.

   ``Buf`` cannot be `NULL`.

   This function should be paired with LLVMRemarkParserDispose() to avoid
   leaking resources.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkParserRef LLVMRemarkParserCreateYAML(const void * Buf, uint64_t Size)


.. py:function:: LLVMRemarkParserCreateBitstream(Buf, Size)

   Creates a remark parser that can be used to parse the buffer located in \p
   Buf of size ``Size`` bytes.

   ``Buf`` cannot be `NULL`.

   This function should be paired with LLVMRemarkParserDispose() to avoid
   leaking resources.

   Since:
       REMARKS_API_VERSION=1

   Args:
       Buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkParserRef LLVMRemarkParserCreateBitstream(const void * Buf, uint64_t Size)


.. py:function:: LLVMRemarkParserGetNext(Parser)

   Returns the next remark in the file.

   The value pointed to by the return value needs to be disposed using a call to
   LLVMRemarkEntryDispose().

   All the entries in the returned value that are of LLVMRemarkStringRef type
   will become invalidated once a call to LLVMRemarkParserDispose is made.

   If the parser reaches the end of the buffer, the return value will be `NULL`.

   In the case of an error, the return value will be `NULL`, and:

   1) LLVMRemarkParserHasError() will return `1`.

   2) LLVMRemarkParserGetErrorMessage() will return a descriptive error
      message.

   An error may occur if:

   1) An argument is invalid.

   2) There is a parsing error. This can occur on things like malformed YAML.

   3) There is a Remark semantic error. This can occur on well-formed files with
      missing or extra fields.

   Here is a quick example of the usage:

   ```
   LLVMRemarkParserRef Parser = LLVMRemarkParserCreateYAML(Buf, Size);
   LLVMRemarkEntryRef Remark = NULL;
   while ((Remark = LLVMRemarkParserGetNext(Parser))) {
      // use Remark
      LLVMRemarkEntryDispose(Remark); // Release memory.
   }
   bool HasError = LLVMRemarkParserHasError(Parser);
   LLVMRemarkParserDispose(Parser);
   ```

   Since:
       REMARKS_API_VERSION=0

   Args:
       Parser (:py:obj:`~.LLVMRemarkOpaqueParser`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRemarkEntryRef LLVMRemarkParserGetNext(LLVMRemarkParserRef Parser)


.. py:function:: LLVMRemarkParserHasError(Parser)

   Returns `1` if the parser encountered an error while parsing the buffer.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Parser (:py:obj:`~.LLVMRemarkOpaqueParser`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMRemarkParserHasError(LLVMRemarkParserRef Parser)


.. py:function:: LLVMRemarkParserGetErrorMessage(Parser)

   Returns a null-terminated string containing an error message.

   In case of no error, the result is `NULL`.

   The memory of the string is bound to the lifetime of ``Parser.`` If
   LLVMRemarkParserDispose() is called, the memory of the string will be
   released.

   Since:
       REMARKS_API_VERSION=0

   Args:
       Parser (:py:obj:`~.LLVMRemarkOpaqueParser`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMRemarkParserGetErrorMessage(LLVMRemarkParserRef Parser)


.. py:function:: LLVMRemarkParserDispose(Parser)

   Releases all the resources used by ``Parser.``

   Since:
       REMARKS_API_VERSION=0

   Args:
       Parser (:py:obj:`~.LLVMRemarkOpaqueParser`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMRemarkParserDispose(LLVMRemarkParserRef Parser)


.. py:function:: LLVMRemarkVersion()

   Returns the version of the remarks library.

   Since:
       REMARKS_API_VERSION=0

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMRemarkVersion()


