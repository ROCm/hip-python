rocm.bindings.llvm.c.lto
========================

.. py:module:: rocm.bindings.llvm.c.lto


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.lto.lto_module_t
   rocm.bindings.llvm.c.lto.lto_code_gen_t
   rocm.bindings.llvm.c.lto.thinlto_code_gen_t
   rocm.bindings.llvm.c.lto.lto_input_t


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.lto.lto_symbol_attributes
   rocm.bindings.llvm.c.lto.lto_debug_model
   rocm.bindings.llvm.c.lto.lto_codegen_model
   rocm.bindings.llvm.c.lto.LLVMOpaqueLTOModule
   rocm.bindings.llvm.c.lto.LLVMOpaqueLTOCodeGenerator
   rocm.bindings.llvm.c.lto.LLVMOpaqueThinLTOCodeGenerator
   rocm.bindings.llvm.c.lto.lto_codegen_diagnostic_severity_t
   rocm.bindings.llvm.c.lto.lto_diagnostic_handler_t
   rocm.bindings.llvm.c.lto.LLVMOpaqueLTOInput
   rocm.bindings.llvm.c.lto.LTOObjectBuffer


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.lto.has_symbol
   rocm.bindings.llvm.c.lto.lto_get_version
   rocm.bindings.llvm.c.lto.lto_get_error_message
   rocm.bindings.llvm.c.lto.lto_module_is_object_file
   rocm.bindings.llvm.c.lto.lto_module_is_object_file_for_target
   rocm.bindings.llvm.c.lto.lto_module_has_objc_category
   rocm.bindings.llvm.c.lto.lto_module_is_object_file_in_memory
   rocm.bindings.llvm.c.lto.lto_module_is_object_file_in_memory_for_target
   rocm.bindings.llvm.c.lto.lto_module_create
   rocm.bindings.llvm.c.lto.lto_module_create_from_memory
   rocm.bindings.llvm.c.lto.lto_module_create_from_memory_with_path
   rocm.bindings.llvm.c.lto.lto_module_create_in_local_context
   rocm.bindings.llvm.c.lto.lto_module_create_in_codegen_context
   rocm.bindings.llvm.c.lto.lto_module_create_from_fd
   rocm.bindings.llvm.c.lto.lto_module_create_from_fd_at_offset
   rocm.bindings.llvm.c.lto.lto_module_dispose
   rocm.bindings.llvm.c.lto.lto_module_get_target_triple
   rocm.bindings.llvm.c.lto.lto_module_set_target_triple
   rocm.bindings.llvm.c.lto.lto_module_get_num_symbols
   rocm.bindings.llvm.c.lto.lto_module_get_symbol_name
   rocm.bindings.llvm.c.lto.lto_module_get_symbol_attribute
   rocm.bindings.llvm.c.lto.lto_module_get_num_asm_undef_symbols
   rocm.bindings.llvm.c.lto.lto_module_get_asm_undef_symbol_name
   rocm.bindings.llvm.c.lto.lto_module_get_linkeropts
   rocm.bindings.llvm.c.lto.lto_module_get_macho_cputype
   rocm.bindings.llvm.c.lto.lto_module_has_ctor_dtor
   rocm.bindings.llvm.c.lto.lto_codegen_set_diagnostic_handler
   rocm.bindings.llvm.c.lto.lto_codegen_create
   rocm.bindings.llvm.c.lto.lto_codegen_create_in_local_context
   rocm.bindings.llvm.c.lto.lto_codegen_dispose
   rocm.bindings.llvm.c.lto.lto_codegen_add_module
   rocm.bindings.llvm.c.lto.lto_codegen_set_module
   rocm.bindings.llvm.c.lto.lto_codegen_set_debug_model
   rocm.bindings.llvm.c.lto.lto_codegen_set_pic_model
   rocm.bindings.llvm.c.lto.lto_codegen_set_cpu
   rocm.bindings.llvm.c.lto.lto_codegen_set_assembler_path
   rocm.bindings.llvm.c.lto.lto_codegen_set_assembler_args
   rocm.bindings.llvm.c.lto.lto_codegen_add_must_preserve_symbol
   rocm.bindings.llvm.c.lto.lto_codegen_write_merged_modules
   rocm.bindings.llvm.c.lto.lto_codegen_compile
   rocm.bindings.llvm.c.lto.lto_codegen_compile_to_file
   rocm.bindings.llvm.c.lto.lto_codegen_optimize
   rocm.bindings.llvm.c.lto.lto_codegen_compile_optimized
   rocm.bindings.llvm.c.lto.lto_api_version
   rocm.bindings.llvm.c.lto.lto_set_debug_options
   rocm.bindings.llvm.c.lto.lto_codegen_debug_options
   rocm.bindings.llvm.c.lto.lto_codegen_debug_options_array
   rocm.bindings.llvm.c.lto.lto_initialize_disassembler
   rocm.bindings.llvm.c.lto.lto_codegen_set_should_internalize
   rocm.bindings.llvm.c.lto.lto_codegen_set_should_embed_uselists
   rocm.bindings.llvm.c.lto.lto_input_create
   rocm.bindings.llvm.c.lto.lto_input_dispose
   rocm.bindings.llvm.c.lto.lto_input_get_num_dependent_libraries
   rocm.bindings.llvm.c.lto.lto_input_get_dependent_library
   rocm.bindings.llvm.c.lto.lto_runtime_lib_symbols_list
   rocm.bindings.llvm.c.lto.thinlto_create_codegen
   rocm.bindings.llvm.c.lto.thinlto_codegen_dispose
   rocm.bindings.llvm.c.lto.thinlto_codegen_add_module
   rocm.bindings.llvm.c.lto.thinlto_codegen_process
   rocm.bindings.llvm.c.lto.thinlto_module_get_num_objects
   rocm.bindings.llvm.c.lto.thinlto_module_get_object
   rocm.bindings.llvm.c.lto.thinlto_module_get_num_object_files
   rocm.bindings.llvm.c.lto.thinlto_module_get_object_file
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_pic_model
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_savetemps_dir
   rocm.bindings.llvm.c.lto.thinlto_set_generated_objects_dir
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cpu
   rocm.bindings.llvm.c.lto.thinlto_codegen_disable_codegen
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_codegen_only
   rocm.bindings.llvm.c.lto.thinlto_debug_options
   rocm.bindings.llvm.c.lto.lto_module_is_thinlto
   rocm.bindings.llvm.c.lto.thinlto_codegen_add_must_preserve_symbol
   rocm.bindings.llvm.c.lto.thinlto_codegen_add_cross_referenced_symbol
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_dir
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_pruning_interval
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_final_cache_size_relative_to_available_space
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_entry_expiration
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_size_bytes
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_size_megabytes
   rocm.bindings.llvm.c.lto.thinlto_codegen_set_cache_size_files


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: lto_symbol_attributes

   Bases: :py:obj:`enum.IntEnum`


   prior to LTO_API_VERSION=3

   Since:


   .. py:attribute:: LTO_SYMBOL_ALIGNMENT_MASK
      :type:  int


   .. py:attribute:: LTO_SYMBOL_PERMISSIONS_MASK
      :type:  int


   .. py:attribute:: LTO_SYMBOL_PERMISSIONS_CODE
      :type:  int


   .. py:attribute:: LTO_SYMBOL_PERMISSIONS_DATA
      :type:  int


   .. py:attribute:: LTO_SYMBOL_PERMISSIONS_RODATA
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_MASK
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_REGULAR
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_TENTATIVE
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_WEAK
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_UNDEFINED
      :type:  int


   .. py:attribute:: LTO_SYMBOL_DEFINITION_WEAKUNDEF
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_MASK
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_INTERNAL
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_HIDDEN
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_PROTECTED
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_DEFAULT
      :type:  int


   .. py:attribute:: LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN
      :type:  int


   .. py:attribute:: LTO_SYMBOL_COMDAT
      :type:  int


   .. py:attribute:: LTO_SYMBOL_ALIAS
      :type:  int


.. py:class:: lto_debug_model

   Bases: :py:obj:`enum.IntEnum`


   prior to LTO_API_VERSION=3

   Since:


   .. py:attribute:: LTO_DEBUG_MODEL_NONE
      :type:  int


   .. py:attribute:: LTO_DEBUG_MODEL_DWARF
      :type:  int


.. py:class:: lto_codegen_model

   Bases: :py:obj:`enum.IntEnum`


   prior to LTO_API_VERSION=3

   Since:


   .. py:attribute:: LTO_CODEGEN_PIC_MODEL_STATIC
      :type:  int


   .. py:attribute:: LTO_CODEGEN_PIC_MODEL_DYNAMIC
      :type:  int


   .. py:attribute:: LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC
      :type:  int


   .. py:attribute:: LTO_CODEGEN_PIC_MODEL_DEFAULT
      :type:  int


.. py:class:: LLVMOpaqueLTOModule(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: lto_module_t

.. py:class:: LLVMOpaqueLTOCodeGenerator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: lto_code_gen_t

.. py:class:: LLVMOpaqueThinLTOCodeGenerator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: thinlto_code_gen_t

.. py:function:: lto_get_version()

   Returns a printable string.

   Since:
       prior to LTO_API_VERSION=3

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_get_version()


.. py:function:: lto_get_error_message()

   Returns the last error string or NULL if last operation was successful.

   Since:
       prior to LTO_API_VERSION=3

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_get_error_message()


.. py:function:: lto_module_is_object_file(path)

   Checks if a file is a loadable object file.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_is_object_file(const char * path)


.. py:function:: lto_module_is_object_file_for_target(path, target_triple_prefix)

   Checks if a file is a loadable object compiled for requested target.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       target_triple_prefix (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_is_object_file_for_target(const char * path, const char * target_triple_prefix)


.. py:function:: lto_module_has_objc_category(mem, length)

   Return true if ``Buffer`` contains a bitcode file with ObjC code (category
   or class) in it.

   Since:
       LTO_API_VERSION=20

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_has_objc_category(const void * mem, size_t length)


.. py:function:: lto_module_is_object_file_in_memory(mem, length)

   Checks if a buffer is a loadable object file.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_is_object_file_in_memory(const void * mem, size_t length)


.. py:function:: lto_module_is_object_file_in_memory_for_target(mem, length, target_triple_prefix)

   Checks if a buffer is a loadable object compiled for requested target.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

       target_triple_prefix (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_is_object_file_in_memory_for_target(const void * mem, size_t length, const char * target_triple_prefix)


.. py:function:: lto_module_create(path)

   Loads an object file from disk.

   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       prior to LTO_API_VERSION=3

   Args:
       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create(const char * path)


.. py:function:: lto_module_create_from_memory(mem, length)

   Loads an object file from memory.

   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_from_memory(const void * mem, size_t length)


.. py:function:: lto_module_create_from_memory_with_path(mem, length, path)

   Loads an object file from memory with an extra path argument.

   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=9

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_from_memory_with_path(const void * mem, size_t length, const char * path)


.. py:function:: lto_module_create_in_local_context(mem, length, path)

   Loads an object file in its own context.

   Loads an object file in its own LLVMContext.  This function call is
   thread-safe.  However, modules created this way should not be merged into an
   lto_code_gen_t using *lto_codegen_add_module().*

   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=11

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_in_local_context(const void * mem, size_t length, const char * path)


.. py:function:: lto_module_create_in_codegen_context(mem, length, path, cg)

   Loads an object file in the codegen context.

   Loads an object file into the same context as ``cg.``  The module is safe to
   add using *lto_codegen_add_module().*

   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=11

   Args:
       mem (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_in_codegen_context(const void * mem, size_t length, const char * path, lto_code_gen_t cg)


.. py:function:: lto_module_create_from_fd(fd, path, file_size)

   Loads an object file from disk.

   The seek point of fd is not preserved.
   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=5

   Args:
       fd (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       file_size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_from_fd(int fd, const char * path, size_t file_size)


.. py:function:: lto_module_create_from_fd_at_offset(fd, path, file_size, map_size, offset)

   Loads an object file from disk.

   The seek point of fd is not preserved.
   Returns NULL on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=5

   Args:
       fd (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       file_size (:py:obj:`~.int`):
           (undocumented)

       map_size (:py:obj:`~.int`):
           (undocumented)

       offset (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_module_t lto_module_create_from_fd_at_offset(int fd, const char * path, size_t file_size, size_t map_size, off_t offset)


.. py:function:: lto_module_dispose(mod)

   Frees all memory internally allocated by the module.

   Upon return the lto_module_t is no longer valid.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_module_dispose(lto_module_t mod)


.. py:function:: lto_module_get_target_triple(mod)

   Returns triple string which the object module was compiled under.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_module_get_target_triple(lto_module_t mod)


.. py:function:: lto_module_set_target_triple(mod, triple)

   Sets triple string with which the object will be codegened.

   Since:
       LTO_API_VERSION=4

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

       triple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_module_set_target_triple(lto_module_t mod, const char * triple)


.. py:function:: lto_module_get_num_symbols(mod)

   Returns the number of symbols in the object module.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int lto_module_get_num_symbols(lto_module_t mod)


.. py:function:: lto_module_get_symbol_name(mod, index)

   Returns the name of the ith symbol in the object module.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_module_get_symbol_name(lto_module_t mod, unsigned int index)


.. py:function:: lto_module_get_symbol_attribute(mod, index)

   Returns the attributes of the ith symbol in the object module.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.lto_symbol_attributes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_symbol_attributes lto_module_get_symbol_attribute(lto_module_t mod, unsigned int index)


.. py:function:: lto_module_get_num_asm_undef_symbols(mod)

   Returns the number of asm undefined symbols in the object module.

   Since:
       prior to LTO_API_VERSION=30

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int lto_module_get_num_asm_undef_symbols(lto_module_t mod)


.. py:function:: lto_module_get_asm_undef_symbol_name(mod, index)

   Returns the name of the ith asm undefined symbol in the object module.

   Since:
       prior to LTO_API_VERSION=30

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_module_get_asm_undef_symbol_name(lto_module_t mod, unsigned int index)


.. py:function:: lto_module_get_linkeropts(mod)

   Returns the module's linker options.

   The linker options may consist of multiple flags. It is the linker's
   responsibility to split the flags using a platform-specific mechanism.

   Since:
       LTO_API_VERSION=16

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_module_get_linkeropts(lto_module_t mod)


.. py:function:: lto_module_get_macho_cputype(mod, out_cputype, out_cpusubtype)

   If targeting mach-o on darwin, this function gets the CPU type and subtype
   that will end up being encoded in the mach-o header.

   These are the values
   that can be found in mach/machine.h.

   ``out_cputype`` and ``out_cpusubtype`` must be non-NULL.

   Returns true on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=27

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

       out_cputype (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

       out_cpusubtype (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_get_macho_cputype(lto_module_t mod, unsigned int * out_cputype, unsigned int * out_cpusubtype)


.. py:function:: lto_module_has_ctor_dtor(mod)

   This function can be used by the linker to check if a given module has
   any constructor or destructor functions.

   Returns true if the module has either the @llvm.global_ctors or the
   @llvm.global_dtors symbol. Otherwise returns false.

   Since:
       LTO_API_VERSION=29

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_has_ctor_dtor(lto_module_t mod)


.. py:class:: lto_codegen_diagnostic_severity_t

   Bases: :py:obj:`enum.IntEnum`


   Diagnostic severity.

   Since:
       LTO_API_VERSION=7


   .. py:attribute:: LTO_DS_ERROR
      :type:  int


   .. py:attribute:: LTO_DS_WARNING
      :type:  int


   .. py:attribute:: LTO_DS_REMARK
      :type:  int


   .. py:attribute:: LTO_DS_NOTE
      :type:  int


.. py:class:: lto_diagnostic_handler_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Diagnostic handler type.

   ``severity`` defines the severity.
   ``diag`` is the actual diagnostic.
   The diagnostic is not prefixed by any of severity keyword, e.g., 'error: '.
   ``ctxt`` is used to pass the context set with the diagnostic handler.

   Since:
       LTO_API_VERSION=7


.. py:function:: lto_codegen_set_diagnostic_handler(arg0, arg1, arg2)

   Set a diagnostic handler and the related context (void *).

   This is more general than lto_get_error_message, as the diagnostic handler
   can be called at anytime within lto.

   Since:
       LTO_API_VERSION=7

   Args:
       arg0 (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.lto_diagnostic_handler_t`/:py:obj:`~.object`):
           (undocumented)

       arg2 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_diagnostic_handler(lto_code_gen_t, lto_diagnostic_handler_t, void *)


.. py:function:: lto_codegen_create()

   Instantiates a code generator.

   Returns NULL on error (check lto_get_error_message() for details).

   All modules added using *lto_codegen_add_module()* must have been created
   in the same context as the codegen.

   Since:
       prior to LTO_API_VERSION=3

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_code_gen_t lto_codegen_create()


.. py:function:: lto_codegen_create_in_local_context()

   Instantiate a code generator in its own context.

   Instantiates a code generator in its own context.  Modules added via \a
   lto_codegen_add_module() must have all been created in the same context,
   using *lto_module_create_in_codegen_context().*

   Since:
       LTO_API_VERSION=11

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_code_gen_t lto_codegen_create_in_local_context()


.. py:function:: lto_codegen_dispose(arg0)

   Frees all code generator and all memory it internally allocated.

   Upon return the lto_code_gen_t is no longer valid.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       arg0 (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_dispose(lto_code_gen_t)


.. py:function:: lto_codegen_add_module(cg, mod)

   Add an object module to the set of modules for which code will be generated.

   Returns true on error (check lto_get_error_message() for details).

   ``cg`` and ``mod`` must both be in the same context.  See \a
   lto_codegen_create_in_local_context() and \a
   lto_module_create_in_codegen_context().

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_add_module(lto_code_gen_t cg, lto_module_t mod)


.. py:function:: lto_codegen_set_module(cg, mod)

   Sets the object module for code generation.

   This will transfer the ownership
   of the module to the code generator.

   ``cg`` and ``mod`` must both be in the same context.

   Since:
       LTO_API_VERSION=13

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_module(lto_code_gen_t cg, lto_module_t mod)


.. py:function:: lto_codegen_set_debug_model(cg, arg1)

   Sets if debug info should be generated.

   Returns true on error (check lto_get_error_message() for details).

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.lto_debug_model`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_set_debug_model(lto_code_gen_t cg, lto_debug_model)


.. py:function:: lto_codegen_set_pic_model(cg, arg1)

   Sets which PIC code model to generated.

   Returns true on error (check lto_get_error_message() for details).

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.lto_codegen_model`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_set_pic_model(lto_code_gen_t cg, lto_codegen_model)


.. py:function:: lto_codegen_set_cpu(cg, cpu)

   Sets the cpu to generate code for.

   Since:
       LTO_API_VERSION=4

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       cpu (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_cpu(lto_code_gen_t cg, const char * cpu)


.. py:function:: lto_codegen_set_assembler_path(cg, path)

   Sets the location of the assembler tool to run.

   If not set, libLTO
   will use gcc to invoke the assembler.

   Since:
       LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_assembler_path(lto_code_gen_t cg, const char * path)


.. py:function:: lto_codegen_set_assembler_args(cg, args, nargs)

   Sets extra arguments that libLTO should pass to the assembler.

   Since:
       LTO_API_VERSION=4

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       args (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       nargs (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_assembler_args(lto_code_gen_t cg, const char ** args, int nargs)


.. py:function:: lto_codegen_add_must_preserve_symbol(cg, symbol)

   Adds to a list of all global symbols that must exist in the final generated
   code.

   If a function is not listed there, it might be inlined into every usage
   and optimized away.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       symbol (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_add_must_preserve_symbol(lto_code_gen_t cg, const char * symbol)


.. py:function:: lto_codegen_write_merged_modules(cg, path)

   Writes a new object file at the specified path that contains the
   merged contents of all modules added so far.

   Returns true on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=5

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_write_merged_modules(lto_code_gen_t cg, const char * path)


.. py:function:: lto_codegen_compile(cg, length)

   Generates code for all added modules into one native object file.

   This calls lto_codegen_optimize then lto_codegen_compile_optimized.

   On success returns a pointer to a generated mach-o/ELF buffer and
   length set to the buffer size.  The buffer is owned by the
   lto_code_gen_t and will be freed when lto_codegen_dispose()
   is called, or lto_codegen_compile() is called again.
   On failure, returns NULL (check lto_get_error_message() for details).

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const void * lto_codegen_compile(lto_code_gen_t cg, size_t * length)


.. py:function:: lto_codegen_compile_to_file(cg, name)

   Generates code for all added modules into one native object file.

   This calls lto_codegen_optimize then lto_codegen_compile_optimized (instead
   of returning a generated mach-o/ELF buffer, it writes to a file).

   The name of the file is written to name. Returns true on error.

   Since:
       LTO_API_VERSION=5

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       name (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_compile_to_file(lto_code_gen_t cg, const char ** name)


.. py:function:: lto_codegen_optimize(cg)

   Runs optimization for the merged module. Returns true on error.

   Since:
       LTO_API_VERSION=12

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_codegen_optimize(lto_code_gen_t cg)


.. py:function:: lto_codegen_compile_optimized(cg, length)

   Generates code for the optimized merged module into one native object file.

   It will not run any IR optimizations on the merged module.

   On success returns a pointer to a generated mach-o/ELF buffer and length set
   to the buffer size.  The buffer is owned by the lto_code_gen_t and will be
   freed when lto_codegen_dispose() is called, or
   lto_codegen_compile_optimized() is called again. On failure, returns NULL
   (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=12

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const void * lto_codegen_compile_optimized(lto_code_gen_t cg, size_t * length)


.. py:function:: lto_api_version()

   Returns the runtime API version.

   Since:
       LTO_API_VERSION=12

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int lto_api_version()


.. py:function:: lto_set_debug_options(options, number)

   Parses options immediately, making them available as early as possible.

   For
   example during executing codegen::InitTargetOptionsFromCodeGenFlags. Since
   parsing shud only happen once, only one of lto_codegen_debug_options or
   lto_set_debug_options should be called.

   This function takes one or more options separated by spaces.
   Warning: passing file paths through this function may confuse the argument
   parser if the paths contain spaces.

   Since:
       LTO_API_VERSION=28

   Args:
       options (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       number (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_set_debug_options(const char *const * options, int number)


.. py:function:: lto_codegen_debug_options(cg, arg1)

   Sets options to help debug codegen bugs.

   Since parsing shud only happen once,
   only one of lto_codegen_debug_options or lto_set_debug_options
   should be called.

   This function takes one or more options separated by spaces.
   Warning: passing file paths through this function may confuse the argument
   parser if the paths contain spaces.

   Since:
       prior to LTO_API_VERSION=3

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_debug_options(lto_code_gen_t cg, const char *)


.. py:function:: lto_codegen_debug_options_array(cg, arg1, number)

   Same as the previous function, but takes every option separately through an
   array.

   Since:
       prior to LTO_API_VERSION=26

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       number (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_debug_options_array(lto_code_gen_t cg, const char *const *, int number)


.. py:function:: lto_initialize_disassembler()

   Initializes LLVM disassemblers.

   FIXME: This doesn't really belong here.

   Since:
       LTO_API_VERSION=5

   .. rubric:: C signature

   .. code-block:: c

       void lto_initialize_disassembler()


.. py:function:: lto_codegen_set_should_internalize(cg, ShouldInternalize)

   Sets if we should run internalize pass during optimization and code
   generation.

   Since:
       LTO_API_VERSION=14

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       ShouldInternalize (:py:obj:`~.bint`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_should_internalize(lto_code_gen_t cg, lto_bool_t ShouldInternalize)


.. py:function:: lto_codegen_set_should_embed_uselists(cg, ShouldEmbedUselists)

   Set whether to embed uselists in bitcode.

   Sets whether *lto_codegen_write_merged_modules()* should embed uselists in
   output bitcode.  This should be turned on for all -save-temps output.

   Since:
       LTO_API_VERSION=15

   Args:
       cg (:py:obj:`~.LLVMOpaqueLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       ShouldEmbedUselists (:py:obj:`~.bint`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_codegen_set_should_embed_uselists(lto_code_gen_t cg, lto_bool_t ShouldEmbedUselists)


.. py:class:: LLVMOpaqueLTOInput(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: lto_input_t

.. py:function:: lto_input_create(buffer, buffer_size, path)

   Creates an LTO input file from a buffer.

   The path
   argument is used for diagnotics as this function
   otherwise does not know which file the given buffer
   is associated with.

   Since:
       LTO_API_VERSION=24

   Args:
       buffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       buffer_size (:py:obj:`~.int`):
           (undocumented)

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_input_t lto_input_create(const void * buffer, size_t buffer_size, const char * path)


.. py:function:: lto_input_dispose(input)

   Frees all memory internally allocated by the LTO input file.

   Upon return the lto_module_t is no longer valid.

   Since:
       LTO_API_VERSION=24

   Args:
       input (:py:obj:`~.LLVMOpaqueLTOInput`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void lto_input_dispose(lto_input_t input)


.. py:function:: lto_input_get_num_dependent_libraries(input)

   Returns the number of dependent library specifiers
   for the given LTO input file.

   Since:
       LTO_API_VERSION=24

   Args:
       input (:py:obj:`~.LLVMOpaqueLTOInput`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int lto_input_get_num_dependent_libraries(lto_input_t input)


.. py:function:: lto_input_get_dependent_library(input, index, size)

   Returns the ith dependent library specifier
   for the given LTO input file.

   The returned
   string is not null-terminated.

   Since:
       LTO_API_VERSION=24

   Args:
       input (:py:obj:`~.LLVMOpaqueLTOInput`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * lto_input_get_dependent_library(lto_input_t input, size_t index, size_t * size)


.. py:function:: lto_runtime_lib_symbols_list(size)

   Returns the list of libcall symbols that can be generated by LTO
   that might not be visible from the symbol table of bitcode files.

   Since:
       prior to LTO_API_VERSION=25

   Args:
       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char *const * lto_runtime_lib_symbols_list(size_t * size)


.. py:class:: LTOObjectBuffer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Type to wrap a single object returned by ThinLTO.

   Since:
       LTO_API_VERSION=18


   .. py:attribute:: Buffer
      :type:  Any


   .. py:attribute:: Size
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:function:: thinlto_create_codegen()

   Instantiates a ThinLTO code generator.

   Returns NULL on error (check lto_get_error_message() for details).

   The ThinLTOCodeGenerator is not intended to be reuse for multiple
   compilation: the model is that the client adds modules to the generator and
   ask to perform the ThinLTO optimizations / codegen, and finally destroys the
   codegenerator.

   Since:
       LTO_API_VERSION=18

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       thinlto_code_gen_t thinlto_create_codegen()


.. py:function:: thinlto_codegen_dispose(cg)

   Frees the generator and all memory it internally allocated.

   Upon return the thinlto_code_gen_t is no longer valid.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_dispose(thinlto_code_gen_t cg)


.. py:function:: thinlto_codegen_add_module(cg, identifier, data, length)

   Add a module to a ThinLTO code generator.

   Identifier has to be unique among
   all the modules in a code generator. The data buffer stays owned by the
   client, and is expected to be available for the entire lifetime of the
   thinlto_code_gen_t it is added to.

   On failure, returns NULL (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       identifier (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       data (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_add_module(thinlto_code_gen_t cg, const char * identifier, const char * data, int length)


.. py:function:: thinlto_codegen_process(cg)

   Optimize and codegen all the modules added to the codegenerator using
   ThinLTO.

   Resulting objects are accessible using thinlto_module_get_object().

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_process(thinlto_code_gen_t cg)


.. py:function:: thinlto_module_get_num_objects(cg)

   Returns the number of object files produced by the ThinLTO CodeGenerator.

   It usually matches the number of input files, but this is not a guarantee of
   the API and may change in future implementation, so the client should not
   assume it.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int thinlto_module_get_num_objects(thinlto_code_gen_t cg)


.. py:function:: thinlto_module_get_object(cg, index)

   Returns a reference to the ith object file produced by the ThinLTO
   CodeGenerator.

   Client should use ``thinlto_module_get_num_objects()`` to get the number of
   available objects.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.LTOObjectBuffer`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LTOObjectBuffer thinlto_module_get_object(thinlto_code_gen_t cg, unsigned int index)


.. py:function:: thinlto_module_get_num_object_files(cg)

   Returns the number of object files produced by the ThinLTO CodeGenerator.

   It usually matches the number of input files, but this is not a guarantee of
   the API and may change in future implementation, so the client should not
   assume it.

   Since:
       LTO_API_VERSION=21

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int thinlto_module_get_num_object_files(thinlto_code_gen_t cg)


.. py:function:: thinlto_module_get_object_file(cg, index)

   Returns the path to the ith object file produced by the ThinLTO
   CodeGenerator.

   Client should use ``thinlto_module_get_num_object_files()`` to get the number
   of available objects.

   Since:
       LTO_API_VERSION=21

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       index (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * thinlto_module_get_object_file(thinlto_code_gen_t cg, unsigned int index)


.. py:function:: thinlto_codegen_set_pic_model(cg, arg1)

   Sets which PIC code model to generate.

   Returns true on error (check lto_get_error_message() for details).

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       arg1 (:py:obj:`~.lto_codegen_model`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t thinlto_codegen_set_pic_model(thinlto_code_gen_t cg, lto_codegen_model)


.. py:function:: thinlto_codegen_set_savetemps_dir(cg, save_temps_dir)

   Sets the path to a directory to use as a storage for temporary bitcode files.

   The intention is to make the bitcode files available for debugging at various
   stage of the pipeline.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       save_temps_dir (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_savetemps_dir(thinlto_code_gen_t cg, const char * save_temps_dir)


.. py:function:: thinlto_set_generated_objects_dir(cg, save_temps_dir)

   Set the path to a directory where to save generated object files.

   This
   path can be used by a linker to request on-disk files instead of in-memory
   buffers. When set, results are available through
   thinlto_module_get_object_file() instead of thinlto_module_get_object().

   Since:
       LTO_API_VERSION=21

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       save_temps_dir (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_set_generated_objects_dir(thinlto_code_gen_t cg, const char * save_temps_dir)


.. py:function:: thinlto_codegen_set_cpu(cg, cpu)

   Sets the cpu to generate code for.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       cpu (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cpu(thinlto_code_gen_t cg, const char * cpu)


.. py:function:: thinlto_codegen_disable_codegen(cg, disable)

   Disable CodeGen, only run the stages till codegen and stop.

   The output will
   be bitcode.

   Since:
       LTO_API_VERSION=19

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       disable (:py:obj:`~.bint`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_disable_codegen(thinlto_code_gen_t cg, lto_bool_t disable)


.. py:function:: thinlto_codegen_set_codegen_only(cg, codegen_only)

   Perform CodeGen only: disable all other stages.

   Since:
       LTO_API_VERSION=19

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       codegen_only (:py:obj:`~.bint`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_codegen_only(thinlto_code_gen_t cg, lto_bool_t codegen_only)


.. py:function:: thinlto_debug_options(options, number)

   Parse -mllvm style debug options.

   Since:
       LTO_API_VERSION=18

   Args:
       options (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       number (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_debug_options(const char *const * options, int number)


.. py:function:: lto_module_is_thinlto(mod)

   Test if a module has support for ThinLTO linking.

   Since:
       LTO_API_VERSION=18

   Args:
       mod (:py:obj:`~.LLVMOpaqueLTOModule`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bool`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       lto_bool_t lto_module_is_thinlto(lto_module_t mod)


.. py:function:: thinlto_codegen_add_must_preserve_symbol(cg, name, length)

   Adds a symbol to the list of global symbols that must exist in the final
   generated code.

   If a function is not listed there, it might be inlined into
   every usage and optimized away. For every single module, the functions
   referenced from code outside of the ThinLTO modules need to be added here.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_add_must_preserve_symbol(thinlto_code_gen_t cg, const char * name, int length)


.. py:function:: thinlto_codegen_add_cross_referenced_symbol(cg, name, length)

   Adds a symbol to the list of global symbols that are cross-referenced between
   ThinLTO files.

   If the ThinLTO CodeGenerator can ensure that every
   references from a ThinLTO module to this symbol is optimized away, then
   the symbol can be discarded.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       length (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_add_cross_referenced_symbol(thinlto_code_gen_t cg, const char * name, int length)


.. py:function:: thinlto_codegen_set_cache_dir(cg, cache_dir)

   Sets the path to a directory to use as a cache storage for incremental build.

   Setting this activates caching.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       cache_dir (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_dir(thinlto_code_gen_t cg, const char * cache_dir)


.. py:function:: thinlto_codegen_set_cache_pruning_interval(cg, interval)

   Sets the cache pruning interval (in seconds).

   A negative value disables the
   pruning. An unspecified default value will be applied, and a value of 0 will
   force prunning to occur.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       interval (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_pruning_interval(thinlto_code_gen_t cg, int interval)


.. py:function:: thinlto_codegen_set_final_cache_size_relative_to_available_space(cg, percentage)

   Sets the maximum cache size that can be persistent across build, in terms of
   percentage of the available space on the disk.

   Set to 100 to indicate
   no limit, 50 to indicate that the cache size will not be left over half the
   available space. A value over 100 will be reduced to 100, a value of 0 will
   be ignored. An unspecified default value will be applied.

   The formula looks like:
    AvailableSpace = FreeSpace + ExistingCacheSize
    NewCacheSize = AvailableSpace * P/100

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       percentage (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_final_cache_size_relative_to_available_space(thinlto_code_gen_t cg, unsigned int percentage)


.. py:function:: thinlto_codegen_set_cache_entry_expiration(cg, expiration)

   Sets the expiration (in seconds) for an entry in the cache.

   An unspecified
   default value will be applied. A value of 0 will be ignored.

   Since:
       LTO_API_VERSION=18

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       expiration (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_entry_expiration(thinlto_code_gen_t cg, unsigned int expiration)


.. py:function:: thinlto_codegen_set_cache_size_bytes(cg, max_size_bytes)

   Sets the maximum size of the cache directory (in bytes).

   A value over the
   amount of available space on the disk will be reduced to the amount of
   available space. An unspecified default value will be applied. A value of 0
   will be ignored.

   Since:
       LTO_API_VERSION=22

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       max_size_bytes (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_size_bytes(thinlto_code_gen_t cg, unsigned int max_size_bytes)


.. py:function:: thinlto_codegen_set_cache_size_megabytes(cg, max_size_megabytes)

   Same as thinlto_codegen_set_cache_size_bytes, except the maximum size is in
   megabytes (2^20 bytes).

   Since:
       LTO_API_VERSION=23

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       max_size_megabytes (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_size_megabytes(thinlto_code_gen_t cg, unsigned int max_size_megabytes)


.. py:function:: thinlto_codegen_set_cache_size_files(cg, max_size_files)

   Sets the maximum number of files in the cache directory.

   An unspecified
   default value will be applied. A value of 0 will be ignored.

   Since:
       LTO_API_VERSION=22

   Args:
       cg (:py:obj:`~.LLVMOpaqueThinLTOCodeGenerator`/:py:obj:`~.object`):
           (undocumented)

       max_size_files (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void thinlto_codegen_set_cache_size_files(thinlto_code_gen_t cg, unsigned int max_size_files)


