rocm.bindings.llvm.c.blake3
===========================

.. py:module:: rocm.bindings.llvm.c.blake3


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.blake3.llvm_blake3_chunk_state
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.blake3.has_symbol
   rocm.bindings.llvm.c.blake3.llvm_blake3_version
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_init
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_init_keyed
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_init_derive_key
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_init_derive_key_raw
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_update
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_finalize
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_finalize_seek
   rocm.bindings.llvm.c.blake3.llvm_blake3_hasher_reset


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: llvm_blake3_chunk_state(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: cv
      :type:  Any


   .. py:attribute:: chunk_counter
      :type:  Any


   .. py:attribute:: buf
      :type:  Any


   .. py:attribute:: buf_len
      :type:  Any


   .. py:attribute:: blocks_compressed
      :type:  Any


   .. py:attribute:: flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: llvm_blake3_hasher(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: key
      :type:  Any


   .. py:attribute:: chunk
      :type:  Any


   .. py:attribute:: cv_stack_len
      :type:  Any


   .. py:attribute:: cv_stack
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: llvm_blake3_version()

   (No short description, might be part of a group.)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * llvm_blake3_version()


.. py:function:: llvm_blake3_hasher_init(self)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_init(llvm_blake3_hasher * self)


.. py:function:: llvm_blake3_hasher_init_keyed(self, key)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       key (:py:obj:`~.l`/:py:obj:`~.i`/:py:obj:`~.s`/:py:obj:`~.t`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_init_keyed(llvm_blake3_hasher * self, const uint8_t[32] key)


.. py:function:: llvm_blake3_hasher_init_derive_key(self, context)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       context (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_init_derive_key(llvm_blake3_hasher * self, const char * context)


.. py:function:: llvm_blake3_hasher_init_derive_key_raw(self, context, context_len)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       context (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       context_len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_init_derive_key_raw(llvm_blake3_hasher * self, const void * context, size_t context_len)


.. py:function:: llvm_blake3_hasher_update(self, input, input_len)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       input (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       input_len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_update(llvm_blake3_hasher * self, const void * input, size_t input_len)


.. py:function:: llvm_blake3_hasher_finalize(self, out, out_len)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       out_len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_finalize(const llvm_blake3_hasher * self, uint8_t * out, size_t out_len)


.. py:function:: llvm_blake3_hasher_finalize_seek(self, seek, out, out_len)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

       seek (:py:obj:`~.int`):
           (undocumented)

       out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       out_len (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_finalize_seek(const llvm_blake3_hasher * self, uint64_t seek, uint8_t * out, size_t out_len)


.. py:function:: llvm_blake3_hasher_reset(self)

   (No short description, might be part of a group.)

   Args:
       self (:py:obj:`~.llvm_blake3_hasher`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void llvm_blake3_hasher_reset(llvm_blake3_hasher * self)


