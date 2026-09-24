rocm.bindings.llvm.c.object
===========================

.. py:module:: rocm.bindings.llvm.c.object


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.object.LLVMSectionIteratorRef
   rocm.bindings.llvm.c.object.LLVMSymbolIteratorRef
   rocm.bindings.llvm.c.object.LLVMRelocationIteratorRef
   rocm.bindings.llvm.c.object.LLVMObjectFileRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.object.LLVMOpaqueSectionIterator
   rocm.bindings.llvm.c.object.LLVMOpaqueSymbolIterator
   rocm.bindings.llvm.c.object.LLVMOpaqueRelocationIterator
   rocm.bindings.llvm.c.object.LLVMBinaryType
   rocm.bindings.llvm.c.object.LLVMOpaqueObjectFile


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.object.has_symbol
   rocm.bindings.llvm.c.object.LLVMCreateBinary
   rocm.bindings.llvm.c.object.LLVMDisposeBinary
   rocm.bindings.llvm.c.object.LLVMBinaryCopyMemoryBuffer
   rocm.bindings.llvm.c.object.LLVMBinaryGetType
   rocm.bindings.llvm.c.object.LLVMMachOUniversalBinaryCopyObjectForArch
   rocm.bindings.llvm.c.object.LLVMObjectFileCopySectionIterator
   rocm.bindings.llvm.c.object.LLVMObjectFileIsSectionIteratorAtEnd
   rocm.bindings.llvm.c.object.LLVMObjectFileCopySymbolIterator
   rocm.bindings.llvm.c.object.LLVMObjectFileIsSymbolIteratorAtEnd
   rocm.bindings.llvm.c.object.LLVMDisposeSectionIterator
   rocm.bindings.llvm.c.object.LLVMMoveToNextSection
   rocm.bindings.llvm.c.object.LLVMMoveToContainingSection
   rocm.bindings.llvm.c.object.LLVMDisposeSymbolIterator
   rocm.bindings.llvm.c.object.LLVMMoveToNextSymbol
   rocm.bindings.llvm.c.object.LLVMGetSectionName
   rocm.bindings.llvm.c.object.LLVMGetSectionSize
   rocm.bindings.llvm.c.object.LLVMGetSectionContents
   rocm.bindings.llvm.c.object.LLVMGetSectionAddress
   rocm.bindings.llvm.c.object.LLVMGetSectionContainsSymbol
   rocm.bindings.llvm.c.object.LLVMGetRelocations
   rocm.bindings.llvm.c.object.LLVMDisposeRelocationIterator
   rocm.bindings.llvm.c.object.LLVMIsRelocationIteratorAtEnd
   rocm.bindings.llvm.c.object.LLVMMoveToNextRelocation
   rocm.bindings.llvm.c.object.LLVMGetSymbolName
   rocm.bindings.llvm.c.object.LLVMGetSymbolAddress
   rocm.bindings.llvm.c.object.LLVMGetSymbolSize
   rocm.bindings.llvm.c.object.LLVMGetRelocationOffset
   rocm.bindings.llvm.c.object.LLVMGetRelocationSymbol
   rocm.bindings.llvm.c.object.LLVMGetRelocationType
   rocm.bindings.llvm.c.object.LLVMGetRelocationTypeName
   rocm.bindings.llvm.c.object.LLVMGetRelocationValueString
   rocm.bindings.llvm.c.object.LLVMCreateObjectFile
   rocm.bindings.llvm.c.object.LLVMDisposeObjectFile
   rocm.bindings.llvm.c.object.LLVMGetSections
   rocm.bindings.llvm.c.object.LLVMIsSectionIteratorAtEnd
   rocm.bindings.llvm.c.object.LLVMGetSymbols
   rocm.bindings.llvm.c.object.LLVMIsSymbolIteratorAtEnd


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpaqueSectionIterator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMSectionIteratorRef

.. py:class:: LLVMOpaqueSymbolIterator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMSymbolIteratorRef

.. py:class:: LLVMOpaqueRelocationIterator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMRelocationIteratorRef

.. py:class:: LLVMBinaryType

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: LLVMBinaryTypeArchive
      :type:  int


   .. py:attribute:: LLVMBinaryTypeMachOUniversalBinary
      :type:  int


   .. py:attribute:: LLVMBinaryTypeCOFFImportFile
      :type:  int


   .. py:attribute:: LLVMBinaryTypeIR
      :type:  int


   .. py:attribute:: LLVMBinaryTypeWinRes
      :type:  int


   .. py:attribute:: LLVMBinaryTypeCOFF
      :type:  int


   .. py:attribute:: LLVMBinaryTypeELF32L
      :type:  int


   .. py:attribute:: LLVMBinaryTypeELF32B
      :type:  int


   .. py:attribute:: LLVMBinaryTypeELF64L
      :type:  int


   .. py:attribute:: LLVMBinaryTypeELF64B
      :type:  int


   .. py:attribute:: LLVMBinaryTypeMachO32L
      :type:  int


   .. py:attribute:: LLVMBinaryTypeMachO32B
      :type:  int


   .. py:attribute:: LLVMBinaryTypeMachO64L
      :type:  int


   .. py:attribute:: LLVMBinaryTypeMachO64B
      :type:  int


   .. py:attribute:: LLVMBinaryTypeWasm
      :type:  int


   .. py:attribute:: LLVMBinaryTypeOffload
      :type:  int


   .. py:attribute:: LLVMBinaryTypeDXcontainer
      :type:  int


.. py:function:: LLVMCreateBinary(MemBuf, Context, ErrorMessage)

   Create a binary file from the given memory buffer.

   The exact type of the binary file will be inferred automatically, and the
   appropriate implementation selected.  The context may be NULL except if
   the resulting file is an LLVM IR file.

   The memory buffer is not consumed by this function. It is the responsibility
   of the caller to free it with ``LLVMDisposeMemoryBuffer.``

   If NULL is returned, the ``ErrorMessage`` parameter is populated with the
   error's description.  It is then the caller's responsibility to free this
   message by calling ``LLVMDisposeMessage.``

   See:
       llvm::object::createBinary

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Context (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ErrorMessage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBinaryRef LLVMCreateBinary(LLVMMemoryBufferRef MemBuf, LLVMContextRef Context, char ** ErrorMessage)


.. py:function:: LLVMDisposeBinary(BR)

   Dispose of a binary file.

   The binary file does not own its backing buffer. It is the responsibility
   of the caller to free it with ``LLVMDisposeMemoryBuffer.``

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeBinary(LLVMBinaryRef BR)


.. py:function:: LLVMBinaryCopyMemoryBuffer(BR)

   Retrieves a copy of the memory buffer associated with this object file.

   The returned buffer is merely a shallow copy and does not own the actual
   backing buffer of the binary. Nevertheless, it is the responsibility of the
   caller to free it with ``LLVMDisposeMemoryBuffer.``

   See:
       llvm::object::getMemoryBufferRef

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMemoryBufferRef LLVMBinaryCopyMemoryBuffer(LLVMBinaryRef BR)


.. py:function:: LLVMBinaryGetType(BR)

   Retrieve the specific type of a binary.

   See:
       llvm::object::Binary::getType

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.LLVMBinaryType`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBinaryType LLVMBinaryGetType(LLVMBinaryRef BR)


.. py:function:: LLVMMachOUniversalBinaryCopyObjectForArch(BR, Arch, ArchLen, ErrorMessage)

   For a Mach-O universal binary file, retrieves the object file corresponding
   to the given architecture if it is present as a slice.

   If NULL is returned, the ``ErrorMessage`` parameter is populated with the
   error's description.  It is then the caller's responsibility to free this
   message by calling ``LLVMDisposeMessage.``

   It is the responsiblity of the caller to free the returned object file by
   calling ``LLVMDisposeBinary.``

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Arch (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       ArchLen (:py:obj:`~.int`):
           (undocumented)

       ErrorMessage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBinaryRef LLVMMachOUniversalBinaryCopyObjectForArch(LLVMBinaryRef BR, const char * Arch, size_t ArchLen, char ** ErrorMessage)


.. py:function:: LLVMObjectFileCopySectionIterator(BR)

   Retrieve a copy of the section iterator for this object file.

   If there are no sections, the result is NULL.

   The returned iterator is merely a shallow copy. Nevertheless, it is
   the responsibility of the caller to free it with
   ``LLVMDisposeSectionIterator.``

   See:
       llvm::object::sections()

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMSectionIteratorRef LLVMObjectFileCopySectionIterator(LLVMBinaryRef BR)


.. py:function:: LLVMObjectFileIsSectionIteratorAtEnd(BR, SI)

   Returns whether the given section iterator is at the end.

   See:
       llvm::object::section_end

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMObjectFileIsSectionIteratorAtEnd(LLVMBinaryRef BR, LLVMSectionIteratorRef SI)


.. py:function:: LLVMObjectFileCopySymbolIterator(BR)

   Retrieve a copy of the symbol iterator for this object file.

   If there are no symbols, the result is NULL.

   The returned iterator is merely a shallow copy. Nevertheless, it is
   the responsibility of the caller to free it with
   ``LLVMDisposeSymbolIterator.``

   See:
       llvm::object::symbols()

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMSymbolIteratorRef LLVMObjectFileCopySymbolIterator(LLVMBinaryRef BR)


.. py:function:: LLVMObjectFileIsSymbolIteratorAtEnd(BR, SI)

   Returns whether the given symbol iterator is at the end.

   See:
       llvm::object::symbol_end

   Args:
       BR (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMObjectFileIsSymbolIteratorAtEnd(LLVMBinaryRef BR, LLVMSymbolIteratorRef SI)


.. py:function:: LLVMDisposeSectionIterator(SI)

   Returns whether the given symbol iterator is at the end.

   See:
       llvm::object::symbol_end

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI)


.. py:function:: LLVMMoveToNextSection(SI)

   Returns whether the given symbol iterator is at the end.

   See:
       llvm::object::symbol_end

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveToNextSection(LLVMSectionIteratorRef SI)


.. py:function:: LLVMMoveToContainingSection(Sect, Sym)

   Returns whether the given symbol iterator is at the end.

   See:
       llvm::object::symbol_end

   Args:
       Sect (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

       Sym (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect, LLVMSymbolIteratorRef Sym)


.. py:function:: LLVMDisposeSymbolIterator(SI)

   // ObjectFile Symbol iterators

   Args:
       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI)


.. py:function:: LLVMMoveToNextSymbol(SI)

   // ObjectFile Symbol iterators

   Args:
       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI)


.. py:function:: LLVMGetSectionName(SI)

   // SectionRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetSectionName(LLVMSectionIteratorRef SI)


.. py:function:: LLVMGetSectionSize(SI)

   // SectionRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI)


.. py:function:: LLVMGetSectionContents(SI)

   // SectionRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetSectionContents(LLVMSectionIteratorRef SI)


.. py:function:: LLVMGetSectionAddress(SI)

   // SectionRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI)


.. py:function:: LLVMGetSectionContainsSymbol(SI, Sym)

   // SectionRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

       Sym (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI, LLVMSymbolIteratorRef Sym)


.. py:function:: LLVMGetRelocations(Section)

   // Section Relocation iterators

   Args:
       Section (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMRelocationIteratorRef LLVMGetRelocations(LLVMSectionIteratorRef Section)


.. py:function:: LLVMDisposeRelocationIterator(RI)

   // Section Relocation iterators

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeRelocationIterator(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMIsRelocationIteratorAtEnd(Section, RI)

   // Section Relocation iterators

   Args:
       Section (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsRelocationIteratorAtEnd(LLVMSectionIteratorRef Section, LLVMRelocationIteratorRef RI)


.. py:function:: LLVMMoveToNextRelocation(RI)

   // Section Relocation iterators

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMoveToNextRelocation(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMGetSymbolName(SI)

   // SymbolRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetSymbolName(LLVMSymbolIteratorRef SI)


.. py:function:: LLVMGetSymbolAddress(SI)

   // SymbolRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI)


.. py:function:: LLVMGetSymbolSize(SI)

   // SymbolRef accessors

   Args:
       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI)


.. py:function:: LLVMGetRelocationOffset(RI)

   // RelocationRef accessors

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetRelocationOffset(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMGetRelocationSymbol(RI)

   // RelocationRef accessors

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMSymbolIteratorRef LLVMGetRelocationSymbol(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMGetRelocationType(RI)

   // RelocationRef accessors

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMGetRelocationType(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMGetRelocationTypeName(RI)

   // following functions.

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetRelocationTypeName(LLVMRelocationIteratorRef RI)


.. py:function:: LLVMGetRelocationValueString(RI)

   // following functions.

   Args:
       RI (:py:obj:`~.LLVMOpaqueRelocationIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMGetRelocationValueString(LLVMRelocationIteratorRef RI)


.. py:class:: LLVMOpaqueObjectFile(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMObjectFileRef

.. py:function:: LLVMCreateObjectFile(MemBuf)

   Deprecated: Use LLVMCreateBinary instead.

   Args:
       MemBuf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf)


.. py:function:: LLVMDisposeObjectFile(ObjectFile)

   Deprecated: Use LLVMDisposeBinary instead.

   Args:
       ObjectFile (:py:obj:`~.LLVMOpaqueObjectFile`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile)


.. py:function:: LLVMGetSections(ObjectFile)

   Deprecated: Use LLVMObjectFileCopySectionIterator instead.

   Args:
       ObjectFile (:py:obj:`~.LLVMOpaqueObjectFile`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile)


.. py:function:: LLVMIsSectionIteratorAtEnd(ObjectFile, SI)

   Deprecated: Use LLVMObjectFileIsSectionIteratorAtEnd instead.

   Args:
       ObjectFile (:py:obj:`~.LLVMOpaqueObjectFile`/:py:obj:`~.object`):
           (undocumented)

       SI (:py:obj:`~.LLVMOpaqueSectionIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile, LLVMSectionIteratorRef SI)


.. py:function:: LLVMGetSymbols(ObjectFile)

   Deprecated: Use LLVMObjectFileCopySymbolIterator instead.

   Args:
       ObjectFile (:py:obj:`~.LLVMOpaqueObjectFile`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile)


.. py:function:: LLVMIsSymbolIteratorAtEnd(ObjectFile, SI)

   Deprecated: Use LLVMObjectFileIsSymbolIteratorAtEnd instead.

   Args:
       ObjectFile (:py:obj:`~.LLVMOpaqueObjectFile`/:py:obj:`~.object`):
           (undocumented)

       SI (:py:obj:`~.LLVMOpaqueSymbolIterator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile, LLVMSymbolIteratorRef SI)


