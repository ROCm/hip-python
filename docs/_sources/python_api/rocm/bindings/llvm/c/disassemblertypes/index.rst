rocm.bindings.llvm.c.disassemblertypes
======================================

.. py:module:: rocm.bindings.llvm.c.disassemblertypes


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.disassemblertypes.LLVMOpInfoCallback
   rocm.bindings.llvm.c.disassemblertypes.LLVMOpInfoSymbol1
   rocm.bindings.llvm.c.disassemblertypes.LLVMOpInfo1
   rocm.bindings.llvm.c.disassemblertypes.LLVMSymbolLookupCallback


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.disassemblertypes.has_symbol


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMOpInfoCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The type for the operand information call back function.

   This is called to
   get the symbolic information for an operand of an instruction.  Typically
   this is from the relocation information, symbol table, etc.  That block of
   information is saved when the disassembler context is created and passed to
   the call back in the DisInfo parameter.  The instruction containing operand
   is at the PC parameter.  For some instruction sets, there can be more than
   one operand with symbolic information.  To determine the symbolic operand
   information for each operand, the bytes for the specific operand in the
   instruction are specified by the Offset parameter and its byte widith is the
   OpSize parameter.  For instructions sets with fixed widths and one symbolic
   operand per instruction, the Offset parameter will be zero and InstSize
   parameter will be the instruction width.  The information is returned in
   TagBuf and is Triple specific with its specific information defined by the
   value of TagType for that Triple.  If symbolic information is returned the
   function * returns 1, otherwise it returns 0.


.. py:class:: LLVMOpInfoSymbol1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The initial support in LLVM MC for the most general form of a relocatable
   expression is "AddSymbol - SubtractSymbol + Offset".

   For some Darwin targets
   this full form is encoded in the relocation information so that AddSymbol and
   SubtractSymbol can be link edited independent of each other.  Many other
   platforms only allow a relocatable expression of the form AddSymbol + Offset
   to be encoded.

   The LLVMOpInfoCallback() for the TagType value of 1 uses the struct
   LLVMOpInfo1.  The value of the relocatable expression for the operand,
   including any PC adjustment, is passed in to the call back in the Value
   field.  The symbolic information about the operand is returned using all
   the fields of the structure with the Offset of the relocatable expression
   returned in the Value field.  It is possible that some symbols in the
   relocatable expression were assembly temporary symbols, for example
   "Ldata - LpicBase + constant", and only the Values of the symbols without
   symbol names are present in the relocation information.  The VariantKind
   type is one of the Target specific :py:obj:`~.defines` below and is used to print
   operands like "_foo@GOT", ":lower16:_foo", etc.


   .. py:attribute:: Present
      :type:  Any


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: Value
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMOpInfo1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: AddSymbol
      :type:  Any


   .. py:attribute:: SubtractSymbol
      :type:  Any


   .. py:attribute:: Value
      :type:  Any


   .. py:attribute:: VariantKind
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMSymbolLookupCallback(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   The type for the symbol lookup function.

   This may be called by the
   disassembler for things like adding a comment for a PC plus a constant
   offset load instruction to use a symbol name instead of a load address value.
   It is passed the block information is saved when the disassembler context is
   created and the ReferenceValue to look up as a symbol.  If no symbol is found
   for the ReferenceValue NULL is returned.  The ReferenceType of the
   instruction is passed indirectly as is the PC of the instruction in
   ReferencePC.  If the output reference can be determined its type is returned
   indirectly in ReferenceType along with ReferenceName if any, or that is set
   to NULL.


