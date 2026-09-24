rocm.bindings.amd_comgr
=======================

.. py:module:: rocm.bindings.amd_comgr


Attributes
----------

.. autoapisummary::

   rocm.bindings.amd_comgr.AMD_COMGR_INTERFACE_VERSION_MAJOR
   rocm.bindings.amd_comgr.AMD_COMGR_INTERFACE_VERSION_MINOR
   rocm.bindings.amd_comgr.amd_comgr_status_t
   rocm.bindings.amd_comgr.amd_comgr_language_t
   rocm.bindings.amd_comgr.amd_comgr_data_kind_t
   rocm.bindings.amd_comgr.amd_comgr_data_t
   rocm.bindings.amd_comgr.amd_comgr_data_set_t
   rocm.bindings.amd_comgr.amd_comgr_action_info_t
   rocm.bindings.amd_comgr.amd_comgr_metadata_node_t
   rocm.bindings.amd_comgr.amd_comgr_symbol_t
   rocm.bindings.amd_comgr.amd_comgr_disassembly_info_t
   rocm.bindings.amd_comgr.amd_comgr_symbolizer_info_t
   rocm.bindings.amd_comgr.amd_comgr_action_kind_t
   rocm.bindings.amd_comgr.amd_comgr_metadata_kind_t
   rocm.bindings.amd_comgr.amd_comgr_symbol_type_t
   rocm.bindings.amd_comgr.amd_comgr_symbol_info_t
   rocm.bindings.amd_comgr.amd_comgr_code_object_info_t
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite_flag_t
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite_options_t


Classes
-------

.. autoapisummary::

   rocm.bindings.amd_comgr.amd_comgr_status_s
   rocm.bindings.amd_comgr.amd_comgr_language_s
   rocm.bindings.amd_comgr.amd_comgr_data_kind_s
   rocm.bindings.amd_comgr.amd_comgr_data_s
   rocm.bindings.amd_comgr.amd_comgr_data_set_s
   rocm.bindings.amd_comgr.amd_comgr_action_info_s
   rocm.bindings.amd_comgr.amd_comgr_metadata_node_s
   rocm.bindings.amd_comgr.amd_comgr_symbol_s
   rocm.bindings.amd_comgr.amd_comgr_disassembly_info_s
   rocm.bindings.amd_comgr.amd_comgr_symbolizer_info_s
   rocm.bindings.amd_comgr.amd_comgr_create_symbolizer_info_anon_funptr_0
   rocm.bindings.amd_comgr.amd_comgr_action_kind_s
   rocm.bindings.amd_comgr.amd_comgr_metadata_kind_s
   rocm.bindings.amd_comgr.amd_comgr_iterate_map_metadata_anon_funptr_0
   rocm.bindings.amd_comgr.amd_comgr_iterate_symbols_anon_funptr_0
   rocm.bindings.amd_comgr.amd_comgr_symbol_type_s
   rocm.bindings.amd_comgr.amd_comgr_symbol_info_s
   rocm.bindings.amd_comgr.amd_comgr_create_disassembly_info_anon_funptr_0
   rocm.bindings.amd_comgr.amd_comgr_create_disassembly_info_anon_funptr_1
   rocm.bindings.amd_comgr.amd_comgr_create_disassembly_info_anon_funptr_2
   rocm.bindings.amd_comgr.code_object_info_s
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite_flag_s
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite_options_s


Functions
---------

.. autoapisummary::

   rocm.bindings.amd_comgr.has_symbol
   rocm.bindings.amd_comgr.amd_comgr_status_string
   rocm.bindings.amd_comgr.amd_comgr_get_version
   rocm.bindings.amd_comgr.amd_comgr_get_isa_count
   rocm.bindings.amd_comgr.amd_comgr_get_isa_name
   rocm.bindings.amd_comgr.amd_comgr_get_isa_metadata
   rocm.bindings.amd_comgr.amd_comgr_create_data
   rocm.bindings.amd_comgr.amd_comgr_release_data
   rocm.bindings.amd_comgr.amd_comgr_get_data_kind
   rocm.bindings.amd_comgr.amd_comgr_set_data
   rocm.bindings.amd_comgr.amd_comgr_set_data_from_file_slice
   rocm.bindings.amd_comgr.amd_comgr_set_data_name
   rocm.bindings.amd_comgr.amd_comgr_get_data
   rocm.bindings.amd_comgr.amd_comgr_get_data_name
   rocm.bindings.amd_comgr.amd_comgr_get_data_isa_name
   rocm.bindings.amd_comgr.amd_comgr_create_symbolizer_info
   rocm.bindings.amd_comgr.amd_comgr_destroy_symbolizer_info
   rocm.bindings.amd_comgr.amd_comgr_symbolize
   rocm.bindings.amd_comgr.amd_comgr_get_data_metadata
   rocm.bindings.amd_comgr.amd_comgr_destroy_metadata
   rocm.bindings.amd_comgr.amd_comgr_create_data_set
   rocm.bindings.amd_comgr.amd_comgr_destroy_data_set
   rocm.bindings.amd_comgr.amd_comgr_data_set_add
   rocm.bindings.amd_comgr.amd_comgr_data_set_remove
   rocm.bindings.amd_comgr.amd_comgr_action_data_count
   rocm.bindings.amd_comgr.amd_comgr_action_data_get_data
   rocm.bindings.amd_comgr.amd_comgr_create_action_info
   rocm.bindings.amd_comgr.amd_comgr_destroy_action_info
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_isa_name
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_isa_name
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_language
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_language
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_option_list
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_option_list_count
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_option_list_item
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_bundle_entry_ids
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_bundle_entry_id_count
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_bundle_entry_id
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_vfs
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_device_lib_linking
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_block_sizes
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_block_sizes_count
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_block_sizes
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_working_directory_path
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_working_directory_path
   rocm.bindings.amd_comgr.amd_comgr_action_info_set_logging
   rocm.bindings.amd_comgr.amd_comgr_action_info_get_logging
   rocm.bindings.amd_comgr.amd_comgr_do_action
   rocm.bindings.amd_comgr.amd_comgr_get_metadata_kind
   rocm.bindings.amd_comgr.amd_comgr_get_metadata_string
   rocm.bindings.amd_comgr.amd_comgr_get_metadata_map_size
   rocm.bindings.amd_comgr.amd_comgr_iterate_map_metadata
   rocm.bindings.amd_comgr.amd_comgr_metadata_lookup
   rocm.bindings.amd_comgr.amd_comgr_get_metadata_list_size
   rocm.bindings.amd_comgr.amd_comgr_index_list_metadata
   rocm.bindings.amd_comgr.amd_comgr_iterate_symbols
   rocm.bindings.amd_comgr.amd_comgr_symbol_lookup
   rocm.bindings.amd_comgr.amd_comgr_symbol_get_info
   rocm.bindings.amd_comgr.amd_comgr_create_disassembly_info
   rocm.bindings.amd_comgr.amd_comgr_destroy_disassembly_info
   rocm.bindings.amd_comgr.amd_comgr_disassemble_instruction
   rocm.bindings.amd_comgr.amd_comgr_demangle_symbol_name
   rocm.bindings.amd_comgr.amd_comgr_populate_mangled_names
   rocm.bindings.amd_comgr.amd_comgr_get_mangled_name
   rocm.bindings.amd_comgr.amd_comgr_populate_name_expression_map
   rocm.bindings.amd_comgr.amd_comgr_map_name_expression_to_symbol_name
   rocm.bindings.amd_comgr.amd_comgr_lookup_code_object
   rocm.bindings.amd_comgr.amd_comgr_map_elf_virtual_address_to_code_object_offset
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite
   rocm.bindings.amd_comgr.amd_comgr_hotswap_rewrite_with_options


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: AMD_COMGR_INTERFACE_VERSION_MAJOR
   :type:  Any

.. py:data:: AMD_COMGR_INTERFACE_VERSION_MINOR
   :type:  Any

.. py:class:: amd_comgr_status_s

   Bases: :py:obj:`enum.IntEnum`


   Status codes.
       


   .. py:attribute:: AMD_COMGR_STATUS_SUCCESS
      :type:  int


   .. py:attribute:: AMD_COMGR_STATUS_ERROR
      :type:  int


   .. py:attribute:: AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
      :type:  int


   .. py:attribute:: AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
      :type:  int


.. py:data:: amd_comgr_status_t

.. py:class:: amd_comgr_language_s

   Bases: :py:obj:`enum.IntEnum`


   The source languages supported by the compiler.
       


   .. py:attribute:: AMD_COMGR_LANGUAGE_NONE
      :type:  int


   .. py:attribute:: AMD_COMGR_LANGUAGE_OPENCL_1_2
      :type:  int


   .. py:attribute:: AMD_COMGR_LANGUAGE_OPENCL_2_0
      :type:  int


   .. py:attribute:: AMD_COMGR_LANGUAGE_HIP
      :type:  int


   .. py:attribute:: AMD_COMGR_LANGUAGE_LLVM_IR
      :type:  int


   .. py:attribute:: AMD_COMGR_LANGUAGE_LAST
      :type:  int


.. py:data:: amd_comgr_language_t

.. py:function:: amd_comgr_status_string(status)

   Query additional information about a status code.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `status` is an invalid status code, or ``status_string`` is NULL.

   Args:
       status (:py:obj:`~.amd_comgr_status_s`) -- *IN*:
           Status code.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               A NUL-terminated string that describes
               the error status.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_status_string(amd_comgr_status_t status, const char ** status_string)


.. py:function:: amd_comgr_get_version()

   Get the version of the code object manager interface
   supported.

   An interface is backwards compatible with an implementation with an
   equal major version, and a greater than or equal minor version.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`:
               Always returns `~.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS`.
       * :py:obj:`~.int`:
               Major version number.
       * :py:obj:`~.int`:
               Minor version number.

   .. rubric:: C signature

   .. code-block:: c

       void amd_comgr_get_version(size_t * major, size_t * minor)


.. py:class:: amd_comgr_data_kind_s

   Bases: :py:obj:`enum.IntEnum`


   The kinds of data supported.
       


   .. py:attribute:: AMD_COMGR_DATA_KIND_UNDEF
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_SOURCE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_INCLUDE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_DIAGNOSTIC
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_LOG
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_BC
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_EXECUTABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_BYTES
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_FATBIN
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_AR
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_BC_BUNDLE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_AR_BUNDLE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_OBJ_BUNDLE
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_SPIRV
      :type:  int


   .. py:attribute:: AMD_COMGR_DATA_KIND_LAST
      :type:  int


.. py:data:: amd_comgr_data_kind_t

.. py:class:: amd_comgr_data_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to a data object.

   Data objects are used to hold the data which is either an input or
   output of a code object manager action.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_data_t

.. py:class:: amd_comgr_data_set_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to an action data object.

   An action data object holds a set of data objects. These can be
   used as inputs to an action, or produced as the result of an
   action.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_data_set_t

.. py:class:: amd_comgr_action_info_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to an action information object.

   An action information object holds all the necessary information,
   excluding the input data objects, required to perform an action.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_action_info_t

.. py:class:: amd_comgr_metadata_node_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to a metadata node.

   A metadata node handle is used to traverse the metadata associated
   with a data node.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_metadata_node_t

.. py:class:: amd_comgr_symbol_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to a machine code object symbol.

   A symbol handle is used to obtain the properties of symbols of a machine code
   object. A symbol handle is invalidated when the data object containing the
   symbol is destroyed.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_symbol_t

.. py:class:: amd_comgr_disassembly_info_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to a disassembly information object.

   A disassembly information object holds all the necessary information,
   excluding the input data, required to perform disassembly.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_disassembly_info_t

.. py:class:: amd_comgr_symbolizer_info_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A handle to a symbolizer information object.

   A symbolizer information object holds all the necessary information
   required to perform symbolization.


   .. py:attribute:: handle
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_symbolizer_info_t

.. py:function:: amd_comgr_get_isa_count()

   Return the number of isa names supported by this version of
   the code object manager library.

   The isa name specifies the instruction set architecture that should
   be used in the actions that involve machine code generation or
   inspection.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `count` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.int`:
               The number of isa names supported.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_isa_count(size_t * count)


.. py:function:: amd_comgr_get_isa_name(index)

   Return the Nth isa name supported by this version of the
   code object manager library.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `index` is greater than the number of isa name supported by this
   version of the code object manager library. ``isa_name`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       index (:py:obj:`~.int`) -- *IN*:
           The index of the isa name to be returned. The
           first isa name is index 0.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               A null terminated string that is the isa name
               being requested.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_isa_name(size_t index, const char ** isa_name)


.. py:function:: amd_comgr_get_isa_metadata(isa_name)

   Get a handle to the metadata of an isa name.

   The structure of the returned metadata is isa name specific and versioned
   with details specified in
   https://llvm.org/docs/AMDGPUUsage.html:py:obj:`~.code`-object-metadata.
   It can include information about the
   limits for resources such as registers and memory addressing.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `name` is NULL or is not an isa name supported by this version of the
   code object manager library. ``metadata`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The isa name to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_metadata_node_s`:
               A handle to the metadata of the isa name. If
               the isa name has no metadata then the returned handle has a kind of
               ``AMD_COMGR_METADATA_KIND_NULL.`` The handle must be destroyed
               using ``amd_comgr_destroy_metadata.``

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_isa_metadata(const char * isa_name, amd_comgr_metadata_node_t * metadata)


.. py:function:: amd_comgr_create_data(kind)

   Create a data object that can hold data of a specified kind.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `kind` is an invalid data kind, or `AMD_COMGR_DATA_KIND_UNDEF`. ``data`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to create the data object as out of resources.

   Args:
       kind (:py:obj:`~.amd_comgr_data_kind_s`) -- *IN*:
           The kind of data the object is intended to hold.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_data_s`:
               A handle to the data object created. Its reference
               count is set to 1.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_create_data(amd_comgr_data_kind_t kind, amd_comgr_data_t * data)


.. py:function:: amd_comgr_release_data(data)

   Indicate that no longer using a data object handle.

   The reference count of the associated data object is
   decremented. If it reaches 0 it is destroyed.

   Note:
       Although this may lead to the destruction of a data object, it is not
       considered a mutation for the purposes of the restrictions described in @ref
       codeobjectmanager.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to release.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_release_data(amd_comgr_data_t data)


.. py:function:: amd_comgr_get_data_kind(data)

   Get the kind of the data object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object. ``kind`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to create the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_data_kind_s`:
               The kind of data the object.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_data_kind(amd_comgr_data_t data, amd_comgr_data_kind_t * kind)


.. py:function:: amd_comgr_set_data(data, size, bytes)

   Set the data content of a data object to the specified
   bytes.

   Any previous value of the data object is overwritten. Any metadata
   associated with the data object is also replaced which invalidates
   all metadata handles to the old metadata.

   Warning:
       This function mutates the data object; see ``codeobjectmanager`` 
       for restrictions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to update.

       size (:py:obj:`~.int`) -- *IN*:
           The number of bytes in the data specified by ``bytes.``

       bytes (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The bytes to set the data object to. The bytes are
           copied into the data object and can be freed after the call.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_set_data(amd_comgr_data_t data, size_t size, const char * bytes)


.. py:function:: amd_comgr_set_data_from_file_slice(data, file_descriptor, offset, size)

   For the given open posix file descriptor, map a slice of the
   file into the data object. The slice is specified by ``offset`` and ``size.``
   Internally this API calls amd_comgr_set_data and resets data object's
   current state.

   Warning:
       This function mutates the data object; see ``codeobjectmanager`` 
       for restrictions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The operation is successful.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is an invalid or
   the map operation failed.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN,OUT*:
           The data object to update.

       file_descriptor (:py:obj:`~.int`) -- *IN*:
           The native file descriptor for an open file.
           The ``file_descriptor`` must not be passed into a system I/O function
           by any other thread while this function is executing.  The offset in
           the file descriptor may be updated based on the requested size and
           underlying platform. The ``file_descriptor`` may be closed immediately
           after this function returns.

       offset (:py:obj:`~.int`) -- *IN*:
           position relative to the start of the file
           specifying the beginning of the slice in ``file_descriptor.``

       size (:py:obj:`~.int`) -- *IN*:
           Size in bytes of the slice.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_set_data_from_file_slice(amd_comgr_data_t data, int file_descriptor, uint64_t offset, uint64_t size)


.. py:function:: amd_comgr_set_data_name(data, name)

   Set the name associated with a data object.

   When compiling, the full name of an include directive is used to
   reference the contents of the include data object with the same
   name. The name may also be used for other data objects in log and
   diagnostic output.

   Warning:
       This function mutates the data object; see ``codeobjectmanager`` 
       for restrictions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to update.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that specifies the name to
           use for the data object. If NULL then the name is set to the empty
           string.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_set_data_name(amd_comgr_data_t data, const char * name)


.. py:function:: amd_comgr_get_data(data, size, bytes)

   Get the data contents, and/or the size of the data
   associated with a data object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``bytes.`` On return, if ``bytes``
           is NULL, set to the size of the data object contents.

       bytes (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` bytes of the
           data object contents is copied. If NULL, no data is copied, and
           only ``size`` is updated (useful in order to find the size of buffer
           required to copy the data).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_data(amd_comgr_data_t data, size_t * size, char * bytes)


.. py:function:: amd_comgr_get_data_name(data, size, name)

   Get the data object name and/or name length.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``name.`` On return, the size of
           the data object name including the terminating null character.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of the
           data object name are copied. If ``name`` is NULL, only ``size`` is updated
           (useful in order to find the size of buffer required to copy the name).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_data_name(amd_comgr_data_t data, size_t * size, char * name)


.. py:function:: amd_comgr_get_data_isa_name(data, size, isa_name)

   Get the data object isa name and/or isa name length.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, has kind `AMD_COMGR_DATA_KIND_UNDEF`, or is not an isa specific
   kind. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``isa_name.`` On return, if `isa_name` is NULL, set to the size of the isa name including the terminating
           null character.

       isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters
           of the isa name are copied. If NULL, no isa name is copied, and
           only ``size`` is updated (useful in order to find the size of buffer
           required to copy the isa name).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_data_isa_name(amd_comgr_data_t data, size_t * size, char * isa_name)


.. py:class:: amd_comgr_create_symbolizer_info_anon_funptr_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: amd_comgr_create_symbolizer_info(code_object, print_symbol_callback)

   Create a symbolizer info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if ``code_object`` is
   invalid or ``print_symbol_callback`` is null.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to create ``symbolizer_info`` as out of resources.

   Args:
       code_object (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object denoting a code object for which
           symbolization should be performed. The kind of this object must be
           ::AMD_COMGR_DATA_KIND_RELOCATABLE, ::AMD_COMGR_DATA_KIND_EXECUTABLE,
           or ::AMD_COMGR_DATA_KIND_BYTES.

       print_symbol_callback (:py:obj:`~.amd_comgr_create_symbolizer_info_anon_funptr_0`/:py:obj:`~.object`) -- *IN*:
           Function called by a successfull
           symbolize query. ``symbol`` is a null-terminated string containing the
           symbolization of the address and ``user_data`` is an arbitary user data.
           The callback does not own ``symbol,`` and it cannot be referenced once
           the callback returns.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_symbolizer_info_s`:
               A handle to the symbolizer info object created.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_create_symbolizer_info(amd_comgr_data_t code_object, void (*)(const char *, void *) print_symbol_callback, amd_comgr_symbolizer_info_t * symbolizer_info)


.. py:function:: amd_comgr_destroy_symbolizer_info(symbolizer_info)

   Destroy symbolizer info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS on successful execution.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if `symbolizer_info` is invalid.

   Args:
       symbolizer_info (:py:obj:`~.amd_comgr_symbolizer_info_s`) -- *IN*:
           A handle to symbolizer info object to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_destroy_symbolizer_info(amd_comgr_symbolizer_info_t symbolizer_info)


.. py:function:: amd_comgr_symbolize(symbolizer_info, address, is_code, user_data)

   Symbolize an address.

   The ``address`` is symbolized using the symbol definitions of the
   ``code_object`` specified when the ``symbolizer_info`` was created.
   The ``print_symbol_callback`` callback function specified when the
   ``symbolizer_info`` was created is called passing the
   symbolization result as ``symbol`` and ``user_data`` value.

   If symbolization is not possible ::AMD_COMGR_STATUS_SUCCESS is returned and
   the string passed to the ``symbol`` argument of the ``print_symbol_callback``
   specified when the ``symbolizer_info`` was created contains the text
   "<invalid>" or "??". This is consistent with `llvm-symbolizer` utility.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `symbolizer_info` is an invalid data object.

   Args:
       symbolizer_info (:py:obj:`~.amd_comgr_symbolizer_info_s`) -- *IN*:
           A handle to symbolizer info object which should be
           used to symbolize the ``address.``

       address (:py:obj:`~.int`) -- *IN*:
           An unrelocated ELF address to which symbolization
           query should be performed.

       is_code (:py:obj:`~.bint`) -- *IN*:
           if true, the symbolizer symbolize the address as code
           and the symbolization result contains filename, function name, line number
           and column number, else the symbolizer symbolize the address as data and
           the symbolizaion result contains symbol name, symbol's starting address
           and symbol size.

       user_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Arbitrary user-data passed to ``print_symbol_callback``
           callback as described for ``symbolizer_info`` argument.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_symbolize(amd_comgr_symbolizer_info_t symbolizer_info, uint64_t address, _Bool is_code, void * user_data)


.. py:function:: amd_comgr_get_data_metadata(data)

   Get a handle to the metadata of a data object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `data` is an invalid data object, or has kind `AMD_COMGR_DATA_KIND_UNDEF`. ``metadata`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_metadata_node_s`:
               A handle to the metadata of the data
               object. If the data object has no metadata then the returned handle
               has a kind of ``AMD_COMGR_METADATA_KIND_NULL.`` The
               handle must be destroyed using ``amd_comgr_destroy_metadata.``

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_data_metadata(amd_comgr_data_t data, amd_comgr_metadata_node_t * metadata)


.. py:function:: amd_comgr_destroy_metadata(metadata)

   Destroy a metadata handle.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``metadata`` is an invalid
   metadata handle.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update metadata
   handle as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           A metadata handle to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_destroy_metadata(amd_comgr_metadata_node_t metadata)


.. py:function:: amd_comgr_create_data_set()

   Create a data set object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to create the data
   set object as out of resources.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_data_set_s`:
               A handle to the data set created. Initially it
               contains no data objects.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_create_data_set(amd_comgr_data_set_t * data_set)


.. py:function:: amd_comgr_destroy_data_set(data_set)

   Destroy a data set object.

   The reference counts of any associated data objects are decremented. Any
   handles to the data set object become invalid.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is an invalid
   data set object.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
   object as out of resources.

   Args:
       data_set (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           A handle to the data set object to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_destroy_data_set(amd_comgr_data_set_t data_set)


.. py:function:: amd_comgr_data_set_add(data_set, data)

   Add a data object to a data set object if it is not already added.

   The reference count of the data object is incremented.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is an invalid
   data set object. ``data`` is an invalid data object; has undef kind; has
   include kind but does not have a name.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
   object as out of resources.

   Args:
       data_set (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           A handle to the data set object to be updated.

       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A handle to the data object to be added. If ``data_set``
           already has the specified handle present, then it is not added. The order
           that data objects are added is preserved.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_data_set_add(amd_comgr_data_set_t data_set, amd_comgr_data_t data)


.. py:function:: amd_comgr_data_set_remove(data_set, data_kind)

   Remove all data objects of a specified kind from a data set object.

   The reference count of the removed data objects is decremented.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is an invalid
   data set object. ``data_kind`` is an invalid data kind.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
   object as out of resources.

   Args:
       data_set (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           A handle to the data set object to be updated.

       data_kind (:py:obj:`~.amd_comgr_data_kind_s`) -- *IN*:
           The data kind of the data objects to be removed. If `AMD_COMGR_DATA_KIND_UNDEF` is specified then all data objects are removed.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_data_set_remove(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind)


.. py:function:: amd_comgr_action_data_count(data_set, data_kind)

   Return the number of data objects of a specified data kind that are
   added to a data set object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is an invalid
   data set object. ``data_kind`` is an invalid data kind or `AMD_COMGR_DATA_KIND_UNDEF`. ``count`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query data set
   object as out of resources.

   Args:
       data_set (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           A handle to the data set object to be queried.

       data_kind (:py:obj:`~.amd_comgr_data_kind_s`) -- *IN*:
           The data kind of the data objects to be counted.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.int`:
               The number of data objects of data kind ``data_kind.``

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_data_count(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t * count)


.. py:function:: amd_comgr_action_data_get_data(data_set, data_kind, index)

   Return the Nth data object of a specified data kind that is added to a
   data set object.

   The reference count of the returned data object is incremented.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data_set`` is an invalid
   data set object. ``data_kind`` is an invalid data kind or `AMD_COMGR_DATA_KIND_UNDEF`. ``index`` is greater than the number of data
   objects of kind ``data_kind.`` ``data`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query data set
   object as out of resources.

   Args:
       data_set (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           A handle to the data set object to be queried.

       data_kind (:py:obj:`~.amd_comgr_data_kind_s`) -- *IN*:
           The data kind of the data object to be returned.

       index (:py:obj:`~.int`) -- *IN*:
           The index of the data object of data kind @data_kind to be
           returned. The first data object is index 0. The order of data objects matches
           the order that they were added to the data set object.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_data_s`:
               The data object being requested.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_data_get_data(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t index, amd_comgr_data_t * data)


.. py:function:: amd_comgr_create_action_info()

   Create an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to create the action info object as out of resources.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_action_info_s`:
               A handle to the action info object created.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_create_action_info(amd_comgr_action_info_t * action_info)


.. py:function:: amd_comgr_destroy_action_info(action_info)

   Destroy an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_destroy_action_info(amd_comgr_action_info_t action_info)


.. py:function:: amd_comgr_action_info_set_isa_name(action_info, isa_name)

   Set the isa name of an action info object.

   When an action info object is created it has no isa name. Some
   actions require that the action info object has an isa name
   defined.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``isa_name`` is not an
   isa name supported by this version of the code object manager
   library.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name. If NULL
           or the empty string then the isa name is cleared. The isa name is defined as
           the Code Object Target Identification string, described at
           https://llvm.org/docs/AMDGPUUsage.html:py:obj:`~.code`-object-target-identification

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_isa_name(amd_comgr_action_info_t action_info, const char * isa_name)


.. py:function:: amd_comgr_action_info_get_isa_name(action_info, size, isa_name)

   Get the isa name and/or isa name length.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``isa_name.`` On return, if `isa_name` is NULL, set to the size of the isa name including the terminating
           null character.

       isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of the
           isa name are copied into ``isa_name.`` If the isa name is not set then an
           empty string is copied into ``isa_name.`` If NULL, no name is copied, and
           only ``size`` is updated (useful in order to find the size of buffer required
           to copy the name).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_isa_name(amd_comgr_action_info_t action_info, size_t * size, char * isa_name)


.. py:function:: amd_comgr_action_info_set_language(action_info, language)

   Set the source language of an action info object.

   When an action info object is created it has no language defined
   which is represented by `AMD_COMGR_LANGUAGE_NONE`. Some actions require that
   the action info object has a source language defined.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``language`` is an
   invalid language.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       language (:py:obj:`~.amd_comgr_language_s`) -- *IN*:
           The language to set. If `AMD_COMGR_LANGUAGE_NONE` then the language is cleared.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_language(amd_comgr_action_info_t action_info, amd_comgr_language_t language)


.. py:function:: amd_comgr_action_info_get_language(action_info, language)

   Get the language for an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``language`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       language (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           The language of the action info opject. `AMD_COMGR_LANGUAGE_NONE` if not defined,

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_language(amd_comgr_action_info_t action_info, amd_comgr_language_t * language)


.. py:function:: amd_comgr_action_info_set_option_list(action_info, options, count)

   Set the options array of an action info object.

   This overrides any option strings or arrays previously set by calls to this
   function.

   An ``action_info`` object which had its options set with this function can
   only have its option inspected with `amd_comgr_action_info_get_option_list_count` and `amd_comgr_action_info_get_option_list_item`.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object, or ``options`` is NULL and ``count`` is non-zero.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update action
   info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be updated.

       options (:py:obj:`~.rocm.bindings.util.types.ListOfBytes`/:py:obj:`~.object`) -- *IN*:
           An array of null terminated strings. May be NULL if `count` is zero, which will result in an empty options array.

       count (:py:obj:`~.int`) -- *IN*:
           The number of null terminated strings in ``options.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_option_list(amd_comgr_action_info_t action_info, const char *[] options, size_t count)


.. py:function:: amd_comgr_action_info_get_option_list_count(action_info)

   Return the number of options in the options array.

   The ``action_info`` object must have had its options set with `amd_comgr_action_info_set_option_list`.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The options of ``action_info`` were never
   set, or not set with ``amd_comgr_action_info_set_option_list.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object, or ``count`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query the data
   object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.int`:
               The number of options in the options array.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_option_list_count(amd_comgr_action_info_t action_info, size_t * count)


.. py:function:: amd_comgr_action_info_get_option_list_item(action_info, index, size, option)

   Return the Nth option string in the options array and/or that
   option's length.

   The ``action_info`` object must have had its options set with `amd_comgr_action_info_set_option_list`.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The options of ``action_info`` were never
   set, or not set with ``amd_comgr_action_info_set_option_list.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object, ``index`` is invalid, or ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query the data
   object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       index (:py:obj:`~.int`) -- *IN*:
           The index of the option to be returned. The first option
           index is 0. The order is the same as the options when they were added in `amd_comgr_action_info_set_option_list`.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``option.`` On return, if @option
           is NULL, set to the size of the Nth option string including the terminating
           null character.

       option (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of the Nth
           option string are copied into ``option.`` If NULL, no option string is
           copied, and only ``size`` is updated (useful in order to find the size of
           buffer required to copy the option string).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_option_list_item(amd_comgr_action_info_t action_info, size_t index, size_t * size, char * option)


.. py:function:: amd_comgr_action_info_set_bundle_entry_ids(action_info, bundle_entry_ids, count)

   Set the bundle entry IDs of an action info object.

   When an action info object is created it has no bundle entry IDs. Some
   actions require that the action info object has bundle entry IDs
   defined.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``contains`` an invalid
   bundle ID not supported by this version of the code object manager
   library.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       bundle_entry_ids (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           An array of strings containing one or more
           bundle entry ID strings. If NULL then the bundle entry ID strings are
           cleared. These IDs are described at
           https://clang.llvm.org/docs/ClangOffloadBundler.html:py:obj:`~.bundle`-entry-id

       count (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_bundle_entry_ids(amd_comgr_action_info_t action_info, const char *[] bundle_entry_ids, size_t count)


.. py:function:: amd_comgr_action_info_get_bundle_entry_id_count(action_info, count)

   Get number of bundle entry IDs

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       count (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The number of bundle entry IDs availible. This value
           can be used as an upper bound to the Index provided to the corresponding
           amd_comgr_get_bundle_entry_id() call.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_bundle_entry_id_count(amd_comgr_action_info_t action_info, size_t * count)


.. py:function:: amd_comgr_action_info_get_bundle_entry_id(action_info, index, size, bundle_entry_id)

   Fetch the Nth specific bundle entry ID or that ID's length.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       index (:py:obj:`~.int`) -- *IN*:
           The index of the bundle entry ID to be returned.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           For out, the size of ``bundle_entry_id.`` For in,
           if @bundle_entry_id is NULL, set to the size of the Nth ID string including
           the terminating null character.

       bundle_entry_id (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of
           the Nth bundle entry ID string are copied into ``bundle_entry_id.`` If NULL,
           no bundle entry ID is copied, and only ``size`` is updated (useful in order
           to find the size of the buffer requried to copy the bundle_entry_id string).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_bundle_entry_id(amd_comgr_action_info_t action_info, size_t index, size_t * size, char * bundle_entry_id)


.. py:function:: amd_comgr_action_info_set_vfs(action_info, should_use_vfs)

   Set whether the specified action should use an
   in-memory virtual file system (VFS).

   Warning:
       Environment variable ``AMD_COMGR_SAVE_TEMPS`` may override options
       set by this API and ``AMD_COMGR_USE_VFS.`` If ``AMD_COMGR_SAVE_TEMPS`` is set
       to "1", all actions are performed using the real file system irrespective of
       the value of ``should_use_vfs`` ``AMD_COMGR_USE_VFS;``

   Warning:
       Environment variable ``AMD_COMGR_USE_VFS`` may override options
       set by this API. If ``AMD_COMGR_USE_VFS`` is set to "1", all actions
       are performed using VFS. If ``AMD_COMGR_USE_VFS`` is set to "0",
       none of the actions are performed using VFS.

   If ``AMD_COMGR_USE_VFS`` is unset, this API can be used to selectively
   turn VFS usage on/off for specified actions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       should_use_vfs (:py:obj:`~.bint`) -- *IN*:
           A boolean that directs the choice to
           use the VFS.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_vfs(amd_comgr_action_info_t action_info, _Bool should_use_vfs)


.. py:function:: amd_comgr_action_info_set_device_lib_linking(action_info, should_link_device_libs)

   Set the device library linking behavior of an action info object.

   Device library linking can be either enforced or omitted for compilation
   actions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       should_link_device_libs (:py:obj:`~.bint`) -- *IN*:
           A boolean that directs the choice to
           link the device libraries.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_device_lib_linking(amd_comgr_action_info_t action_info, _Bool should_link_device_libs)


.. py:function:: amd_comgr_action_info_set_block_sizes(action_info, block_sizes, count)

   Set the block sizes for kernel cloning.

   When an action info object is created it has no block sizes specified.
   When block sizes are set, SPIR-V translation and compilation actions
   (AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC and
   AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE) will automatically clone
   kernel functions for each specified block size, if the size is within the
   limits specified by the amdgpu-flat-work-group-size attribute on the kernel.
   The upper limit of the original kernel is assumed to represent the original
   kernel, so if corresponding block size is in the set of block sizes, no new
   kernel is generated for that block size.
   Cloned kernels will have the amdgpu-flat-work-group-size attribute set with
   an upper limit equal to the corresponding block size and the lower bound
   equal to the lower bound of the original kernel.
   Cloned kernels will have a name in the format of
   "<original_kernel_name>.bs<block_size>".

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object, or ``block_sizes`` is NULL and ``count`` is
   non-zero.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update action
   info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be updated.

       block_sizes (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN*:
           An array of block sizes (flat work group sizes) to
           compile kernel variants for. If NULL then the block sizes are cleared.

       count (:py:obj:`~.int`) -- *IN*:
           The number of elements in the ``block_sizes`` array.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_block_sizes(amd_comgr_action_info_t action_info, const size_t * block_sizes, size_t count)


.. py:function:: amd_comgr_action_info_get_block_sizes_count(action_info, count)

   Get the number of block sizes set for kernel cloning actions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object. ``count`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query action
   info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       count (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The number of block sizes set.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_block_sizes_count(amd_comgr_action_info_t action_info, size_t * count)


.. py:function:: amd_comgr_action_info_get_block_sizes(action_info, count, block_sizes)

   Get the block sizes set for kernel cloning actions.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``action_info`` is an
   invalid action info object. ``block_sizes`` is NULL. ``count`` is less than
   the number of block sizes set.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query action
   info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       count (:py:obj:`~.int`) -- *IN*:
           The number of elements in the ``block_sizes`` array.

       block_sizes (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           Array to store the block sizes. Must be large enough
           to hold ``count`` elements. Use amd_comgr_action_info_get_block_sizes_count
           to query the required size.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_block_sizes(amd_comgr_action_info_t action_info, size_t count, size_t * block_sizes)


.. py:function:: amd_comgr_action_info_set_working_directory_path(action_info, path)

   Set the working directory of an action info object.

   When an action info object is created it has an empty working
   directory. Some actions use the working directory to resolve
   relative file paths.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action info object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the working
           directory path. If NULL or the empty string then the working
           directory is cleared.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_working_directory_path(amd_comgr_action_info_t action_info, const char * path)


.. py:function:: amd_comgr_action_info_get_working_directory_path(action_info, size, path)

   Get the working directory path and/or working directory path
   length of an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``path.`` On return, if ``path`` is
           NULL, set to the size of the working directory path including the
           terminating null character.

       path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of
           the working directory path is copied. If the working directory path
           is not set then an empty string is copied. If NULL, the working
           directory path is not copied, and only ``size`` is updated (useful
           in order to find the size of buffer required to copy the working
           directory path).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_working_directory_path(amd_comgr_action_info_t action_info, size_t * size, char * path)


.. py:function:: amd_comgr_action_info_set_logging(action_info, logging)

   Set whether logging is enabled for an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           A handle to the action info object to be
           updated.

       logging (:py:obj:`~.bint`) -- *IN*:
           Whether logging should be enabled or disable.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_set_logging(amd_comgr_action_info_t action_info, _Bool logging)


.. py:function:: amd_comgr_action_info_get_logging(action_info, logging)

   Get whether logging is enabled for an action info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `action_info` is an invalid action info object. ``logging`` is NULL.

   Args:
       action_info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info object to query.

       logging (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Whether logging is enabled.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_action_info_get_logging(amd_comgr_action_info_t action_info, _Bool * logging)


.. py:class:: amd_comgr_action_kind_s

   Bases: :py:obj:`enum.IntEnum`


   The kinds of actions that can be performed.
       


   .. py:attribute:: AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_LINK_BC_TO_BC
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_UNBUNDLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_COMPILE_SOURCE_TO_SPIRV
      :type:  int


   .. py:attribute:: AMD_COMGR_ACTION_LAST
      :type:  int


.. py:data:: amd_comgr_action_kind_t

.. py:function:: amd_comgr_do_action(kind, info, input, result)

   Perform an action.

   Each action ignores any data objects in ``input`` that it does not
   use. If logging is enabled in @info then ``result`` will have a log
   data object added. Any diagnostic data objects produced by the
   action will be added to ``result.`` See the description of each
   action in ``amd_comgr_action_kind_t.``

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR An error was
   reported when executing the action.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `kind` is an invalid action kind. ``input_data`` or ``result_data`` are
   invalid action data object handles. See the description of each
   action in ``amd_comgr_action_kind_t`` for other
   conditions that result in this status.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       kind (:py:obj:`~.amd_comgr_action_kind_s`) -- *IN*:
           The action to perform.

       info (:py:obj:`~.amd_comgr_action_info_s`) -- *IN*:
           The action info to use when performing the action.

       input (:py:obj:`~.amd_comgr_data_set_s`) -- *IN*:
           The input data objects to the ``kind`` action.

       result (:py:obj:`~.amd_comgr_data_set_s`) -- *OUT*:
           Any data objects are removed before performing
           the action which then adds all data objects produced by the action.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_do_action(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info, amd_comgr_data_set_t input, amd_comgr_data_set_t result)


.. py:class:: amd_comgr_metadata_kind_s

   Bases: :py:obj:`enum.IntEnum`


   The kinds of metadata nodes.
       


   .. py:attribute:: AMD_COMGR_METADATA_KIND_NULL
      :type:  int


   .. py:attribute:: AMD_COMGR_METADATA_KIND_STRING
      :type:  int


   .. py:attribute:: AMD_COMGR_METADATA_KIND_MAP
      :type:  int


   .. py:attribute:: AMD_COMGR_METADATA_KIND_LIST
      :type:  int


   .. py:attribute:: AMD_COMGR_METADATA_KIND_LAST
      :type:  int


.. py:data:: amd_comgr_metadata_kind_t

.. py:function:: amd_comgr_get_metadata_kind(metadata)

   Get the kind of the metadata node.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node. ``kind`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to create the data object as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_metadata_kind_s`:
               The kind of the metadata node.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_metadata_kind(amd_comgr_metadata_node_t metadata, amd_comgr_metadata_kind_t * kind)


.. py:function:: amd_comgr_get_metadata_string(metadata, size, string)

   Get the string and/or string length from a metadata string
   node.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node, or does not have kind `AMD_COMGR_METADATA_KIND_STRING`. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           On entry, the size of ``string.`` On return, if `string` is NULL, set to the size of the string including the terminating null
           character.

       string (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters
           of the string are copied. If NULL, no string is copied, and only `size` is updated (useful in order to find the size of buffer required
           to copy the string).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_metadata_string(amd_comgr_metadata_node_t metadata, size_t * size, char * string)


.. py:function:: amd_comgr_get_metadata_map_size(metadata)

   Get the map size from a metadata map node.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node, or not of kind `AMD_COMGR_METADATA_KIND_MAP`. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.int`:
               The number of entries in the map.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_metadata_map_size(amd_comgr_metadata_node_t metadata, size_t * size)


.. py:class:: amd_comgr_iterate_map_metadata_anon_funptr_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: amd_comgr_iterate_map_metadata(metadata, callback, user_data)

   Iterate over the elements a metadata map node.

   Warning:
       The metadata nodes which are passed to the callback are not owned
       by the callback, and are freed just after the callback returns. The callback
       must not save any references to its parameters between iterations.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR An error was
   reported by ``callback.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node, or not of kind `AMD_COMGR_METADATA_KIND_MAP`. ``callback`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to iterate the metadata as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

       callback (:py:obj:`~.amd_comgr_iterate_map_metadata_anon_funptr_0`/:py:obj:`~.object`) -- *IN*:
           The function to call for each entry in the map. The
           entry's key is passed in ``key,`` the entry's value is passed in ``value,`` and
           ``user_data`` is passed as ``user_data.`` If the function returns with a status
           other than ``AMD_COMGR_STATUS_SUCCESS`` then iteration is stopped.

       user_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The value to pass to each invocation of `callback`. Allows context to be passed into the call back function.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_iterate_map_metadata(amd_comgr_metadata_node_t metadata, amd_comgr_status_t (*)(amd_comgr_metadata_node_t, amd_comgr_metadata_node_t, void *) callback, void * user_data)


.. py:function:: amd_comgr_metadata_lookup(metadata, key)

   Use a string key to lookup an element of a metadata map
   node and return the entry value.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The map has no entry
   with a string key with the value ``key.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node, or not of kind `AMD_COMGR_METADATA_KIND_MAP`. ``key`` or ``value`` is
   NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to lookup metadata as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

       key (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the key to lookup.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_metadata_node_s`:
               The metadata node of the ``key`` element of the
               ``metadata`` map metadata node. The handle must be destroyed
               using ``amd_comgr_destroy_metadata.``

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_metadata_lookup(amd_comgr_metadata_node_t metadata, const char * key, amd_comgr_metadata_node_t * value)


.. py:function:: amd_comgr_get_metadata_list_size(metadata)

   Get the list size from a metadata list node.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node, or does nopt have kind `AMD_COMGR_METADATA_KIND_LIST`. ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update the data object as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.int`:
               The number of entries in the list.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_metadata_list_size(amd_comgr_metadata_node_t metadata, size_t * size)


.. py:function:: amd_comgr_index_list_metadata(metadata, index)

   Return the Nth metadata node of a list metadata node.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT `metadata` is an invalid metadata node or not of kind `AMD_COMGR_METADATA_INFO_LIST`. ``index`` is greater
   than the number of list elements. ``value`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to update action data object as out of resources.

   Args:
       metadata (:py:obj:`~.amd_comgr_metadata_node_s`) -- *IN*:
           The metadata node to query.

       index (:py:obj:`~.int`) -- *IN*:
           The index being requested. The first list element
           is index 0.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_metadata_node_s`:
               The metadata node of the ``index`` element of the
               ``metadata`` list metadata node. The handle must be destroyed
               using ``amd_comgr_destroy_metadata.``

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_index_list_metadata(amd_comgr_metadata_node_t metadata, size_t index, amd_comgr_metadata_node_t * value)


.. py:class:: amd_comgr_iterate_symbols_anon_funptr_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: amd_comgr_iterate_symbols(data, callback, user_data)

   Iterate over the symbols of a machine code object.

   For a AMD_COMGR_DATA_KIND_RELOCATABLE the symbols in the ELF symtab section
   are iterated. For a AMD_COMGR_DATA_KIND_EXECUTABLE the symbols in the ELF
   dynsymtab are iterated.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR An error was
   reported by ``callback.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is an invalid data
   object, or not of kind ``AMD_COMGR_DATA_KIND_RELOCATABLE`` or
   AMD_COMGR_DATA_KIND_EXECUTABLE. ``callback`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to iterate the data object as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

       callback (:py:obj:`~.amd_comgr_iterate_symbols_anon_funptr_0`/:py:obj:`~.object`) -- *IN*:
           The function to call for each symbol in the machine code
           data object. The symbol handle is passed in ``symbol`` and ``user_data`` is
           passed as ``user_data.`` If the function returns with a status other than `AMD_COMGR_STATUS_SUCCESS` then iteration is stopped.

       user_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The value to pass to each invocation of `callback`. Allows context to be passed into the call back function.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_iterate_symbols(amd_comgr_data_t data, amd_comgr_status_t (*)(amd_comgr_symbol_t, void *) callback, void * user_data)


.. py:function:: amd_comgr_symbol_lookup(data, name, symbol)

   Lookup a symbol in a machine code object by name.

   For a AMD_COMGR_DATA_KIND_RELOCATABLE the symbols in the ELF symtab section
   are inspected. For a AMD_COMGR_DATA_KIND_EXECUTABLE the symbols in the ELF
   dynsymtab are inspected.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The machine code object has no symbol
   with ``name.``

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is an invalid data
   object, or not of kind ``AMD_COMGR_DATA_KIND_RELOCATABLE`` or
   AMD_COMGR_DATA_KIND_EXECUTABLE.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to lookup symbol as out of resources.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to query.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the symbol name to lookup.

       symbol (:py:obj:`~.amd_comgr_symbol_s`/:py:obj:`~.object`) -- *OUT*:
           The symbol with the ``name.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_symbol_lookup(amd_comgr_data_t data, const char * name, amd_comgr_symbol_t * symbol)


.. py:class:: amd_comgr_symbol_type_s

   Bases: :py:obj:`enum.IntEnum`


   Machine code object symbol type.
       


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_UNKNOWN
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_NOTYPE
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_OBJECT
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_FUNC
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_SECTION
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_FILE
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_COMMON
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL
      :type:  int


.. py:data:: amd_comgr_symbol_type_t

.. py:class:: amd_comgr_symbol_info_s

   Bases: :py:obj:`enum.IntEnum`


   Machine code object symbol attributes.
       


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_NAME_LENGTH
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_NAME
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_TYPE
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_SIZE
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_VALUE
      :type:  int


   .. py:attribute:: AMD_COMGR_SYMBOL_INFO_LAST
      :type:  int


.. py:data:: amd_comgr_symbol_info_t

.. py:function:: amd_comgr_symbol_get_info(symbol, attribute, value)

   Query information about a machine code object symbol.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has
   been executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The ``symbol`` does not have the requested `attribute`.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``symbol`` is an invalid
   symbol. ``attribute`` is an invalid value. ``value`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
   Unable to query symbol as out of resources.

   Args:
       symbol (:py:obj:`~.amd_comgr_symbol_s`) -- *IN*:
           The symbol to query.

       attribute (:py:obj:`~.amd_comgr_symbol_info_s`) -- *IN*:
           Attribute to query.

       value (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to an application-allocated buffer where to store
           the value of the attribute. If the buffer passed by the application is not
           large enough to hold the value of attribute, the behavior is undefined. The
           type of value returned is specified by ``amd_comgr_symbol_info_t.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_symbol_get_info(amd_comgr_symbol_t symbol, amd_comgr_symbol_info_t attribute, void * value)


.. py:class:: amd_comgr_create_disassembly_info_anon_funptr_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: amd_comgr_create_disassembly_info_anon_funptr_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:class:: amd_comgr_create_disassembly_info_anon_funptr_2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:function:: amd_comgr_create_disassembly_info(isa_name, read_memory_callback, print_instruction_callback, print_address_annotation_callback)

   Create a disassembly info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly info object was created.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``isa_name`` is NULL or
   invalid; or ``read_memory_callback,`` ``print_instruction_callback,``
   or ``print_address_annotation_callback`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to create the
   disassembly info object as out of resources.

   Args:
       isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name of the
           target to disassemble for. The isa name is defined as the Code Object Target
           Identification string, described at
           https://llvm.org/docs/AMDGPUUsage.html:py:obj:`~.code`-object-target-identification

       read_memory_callback (:py:obj:`~.amd_comgr_create_disassembly_info_anon_funptr_0`/:py:obj:`~.object`) -- *IN*:
           Function called to request ``size`` bytes
           from the program address space at ``from`` be read into ``to.`` The requested
           ``size`` is never zero. Returns the number of bytes which could be read, with
           the guarantee that no additional bytes will be available in any subsequent
           call.

       print_instruction_callback (:py:obj:`~.amd_comgr_create_disassembly_info_anon_funptr_1`/:py:obj:`~.object`) -- *IN*:
           Function called after a successful
           disassembly. ``instruction`` is a null terminated string containing the
           disassembled instruction. The callback does not own ``instruction,`` and it
           cannot be referenced once the callback returns.

       print_address_annotation_callback (:py:obj:`~.amd_comgr_create_disassembly_info_anon_funptr_2`/:py:obj:`~.object`) -- *IN*:
           Function called after `print_instruction_callback` returns, once for each instruction operand which
           was resolved to an absolute address. ``address`` is the absolute address in
           the program address space. It is intended to append a symbolic
           form of the address, perhaps as a comment, after the instruction disassembly
           produced by ``print_instruction_callback.``

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)
       * :py:obj:`~.amd_comgr_disassembly_info_s`:
               A handle to the disassembly info object
               created.

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_create_disassembly_info(const char * isa_name, uint64_t (*)(uint64_t, char *, uint64_t, void *) read_memory_callback, void (*)(const char *, void *) print_instruction_callback, void (*)(uint64_t, void *) print_address_annotation_callback, amd_comgr_disassembly_info_t * disassembly_info)


.. py:function:: amd_comgr_destroy_disassembly_info(disassembly_info)

   Destroy a disassembly info object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly info object was
   destroyed.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``disassembly_info`` is an
   invalid disassembly info object.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to destroy the
   disassembly info object as out of resources.

   Args:
       disassembly_info (:py:obj:`~.amd_comgr_disassembly_info_s`) -- *IN*:
           A handle to the disassembly info object to
           destroy.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_destroy_disassembly_info(amd_comgr_disassembly_info_t disassembly_info)


.. py:function:: amd_comgr_disassemble_instruction(disassembly_info, address, user_data, size)

   Disassemble a single instruction.

   @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly was successful.

   @retval ::AMD_COMGR_STATUS_ERROR The disassembly failed.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``disassembly_info`` is
   invalid or ``size`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to disassemble the
   instruction as out of resources.

   Args:
       disassembly_info (:py:obj:`~.amd_comgr_disassembly_info_s`):
           (undocumented)

       address (:py:obj:`~.int`) -- *IN*:
           The address of the first byte of the instruction in the
           program address space.

       user_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Arbitrary user-data passed to each callback function
           during disassembly.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The number of bytes consumed to decode the
           instruction, or consumed while failing to decode an invalid instruction.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_disassemble_instruction(amd_comgr_disassembly_info_t disassembly_info, uint64_t address, void * user_data, uint64_t * size)


.. py:function:: amd_comgr_demangle_symbol_name(mangled_symbol_name, demangled_symbol_name)

   Demangle a symbol name.

   Note:
       If the ``mangled_symbol_name`` cannot be demangled, it will be copied
       without changes to the ``demangled_symbol_name`` and AMD_COMGR_STATUS_SUCCESS
       is returned.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``mangled_symbol_name`` is
   an invalid data object or not of kind ``AMD_COMGR_DATA_KIND_BYTES`` or
   ``demangled_symbol_name`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Out of resources.

   Args:
       mangled_symbol_name (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind `AMD_COMGR_DATA_KIND_BYTES` containing the mangled symbol name.

       demangled_symbol_name (:py:obj:`~.amd_comgr_data_s`/:py:obj:`~.object`) -- *OUT*:
           A handle to the data object of kind `AMD_COMGR_DATA_KIND_BYTES` created and set to contain the demangled symbol
           name in case of successful completion. The handle must be released using
           ``amd_comgr_release_data.`` ``demangled_symbol_name`` is not updated for
           an error case.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_demangle_symbol_name(amd_comgr_data_t mangled_symbol_name, amd_comgr_data_t * demangled_symbol_name)


.. py:function:: amd_comgr_populate_mangled_names(data, count)

   Fetch mangled symbol names from a code object.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is
   an invalid data object or not of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE`` or
   ``AMD_COMGR_DATA_KIND_BC.``

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` or ``AMD_COMGR_DATA_KIND_BC``

       count (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The number of mangled names retrieved. This value
           can be used as an upper bound to the Index provided to the corresponding
           amd_comgr_get_mangled_name() call.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_populate_mangled_names(amd_comgr_data_t data, size_t * count)


.. py:function:: amd_comgr_get_mangled_name(data, index, size, mangled_name)

   Fetch the Nth specific mangled name from a set of populated names or
   that name's length.

   The ``data`` must have had its mangled names populated with `amd_comgr_populate_mangled_names`.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR ``data`` has not been used to
   populate a set of mangled names, or index is greater than the count of
   mangled names for that data object

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` or ``AMD_COMGR_DATA_KIND_BC`` used to
           identify which set of mangled names to retrive from.

       index (:py:obj:`~.int`) -- *IN*:
           The index of the mangled name to be returned.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           For out, the size of ``mangled_name.`` For in,
           if @mangled_name is NULL, set to the size of the Nth option string including
           the terminating null character.

       mangled_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of
           the Nth mangled name string are copied into ``mangled_name.`` If NULL, no
           mangled name string is copied, and only ``size`` is updated (useful in order
           to find the size of the buffer requried to copy the mangled_name string).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_get_mangled_name(amd_comgr_data_t data, size_t index, size_t * size, char * mangled_name)


.. py:function:: amd_comgr_populate_name_expression_map(data, count)

   Populate a name expression map from a given code object.

   Used to map stub names *__amdgcn_name_expr_* in bitcodes and code
   objects generated by hip runtime to an associated (unmangled) name
   expression and (mangled) symbol name.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is
   an invalid data object or not of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE`` or
   ``AMD_COMGR_DATA_KIND_BC.``

   @retval ::AMD_COMGR_STATUS_ERROR LLVM API failure, which should be
   accompanied by an LLVM error message to stderr

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` or ``AMD_COMGR_DATA_KIND_BC``

       count (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The number of name expressions mapped. This value
           can be used as an upper bound to the Index provided to the corresponding
           amd_comgr_map_name_expression_to_symbol_name() call.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_populate_name_expression_map(amd_comgr_data_t data, size_t * count)


.. py:function:: amd_comgr_map_name_expression_to_symbol_name(data, size, name_expression, symbol_name)

   Fetch a related symbol name for a given name expression;
   or that name's length.

   The ``data`` must have had its name expression map populated with `amd_comgr_populate_name_expression_map`.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.

   @retval ::AMD_COMGR_STATUS_ERROR ``data`` object is not valid (NULL or not of
   type bitcode or code object)

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``name_expression`` is not
   present in the name expression map.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` or ``AMD_COMGR_DATA_KIND_BC`` used to
           identify which map of name expressions to retrieve from.

       size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN,OUT*:
           For out, the size of ``symbol_name.`` For in,
           if @symbol_name is NULL, set to the size of the Nth option string including
           the terminating null character.

       name_expression (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A character array of a name expression. This name
           is used as the key to the name expression map in order to locate the desired
           @symbol_name.

       symbol_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           If not NULL, then the first ``size`` characters of
           the symbol name string mapped from @name_expression are copied into `symbol_name`. If NULL, no symbol name string is copied, and only ``size`` is
           updated (useful in order to find the size of the buffer required to copy the
           symbol_name string).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_map_name_expression_to_symbol_name(amd_comgr_data_t data, size_t * size, const char * name_expression, char * symbol_name)


.. py:class:: code_object_info_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A data structure for Code object information.
       


   .. py:attribute:: isa
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: offset
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: amd_comgr_code_object_info_t

.. py:function:: amd_comgr_lookup_code_object(data, info_list, info_list_size)

   @ brief Given a bundled code object and list of target id strings, extract
   correponding code object information.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The code object bundle header is incorrect
   or reading bundle entries failed.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is not of
   kind AMD_COMGR_DATA_KIND_FATBIN, or AMD_COMGR_DATA_KIND_BYTES or
   AMD_COMGR_DATA_KIND_EXECUTABLE or either ``info_list`` is NULL.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if the ``data`` has
   invalid data.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object for bundled code object. This should be
           of kind AMD_COMGR_DATA_KIND_FATBIN or AMD_COMGR_DATA_KIND_EXECUTABLE or
           AMD_COMGR_DATA_KIND_BYTES. The API interprets the data object of kind
           AMD_COMGR_DATA_KIND_FATBIN as a clang offload bundle and of kind
           AMD_COMGR_DATA_KIND_EXECUTABLE as an executable shared object. For a data
           object of type AMD_COMGR_DATA_KIND_BYTES the API first inspects the data
           passed to determine if it is a fatbin or an executable and performs
           the lookup.

       info_list (:py:obj:`~.code_object_info_s`/:py:obj:`~.object`) -- *IN,OUT*:
           A list of code object information structure
           initialized with null terminated target id strings. If the target id
           is matched in the code object bundle the corresponding code object
           information is updated with offset and size of the code object. If the
           target id is not found the offset and size are set to 0.

       info_list_size (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_lookup_code_object(amd_comgr_data_t data, amd_comgr_code_object_info_t * info_list, size_t info_list_size)


.. py:function:: amd_comgr_map_elf_virtual_address_to_code_object_offset(data, elf_virtual_address, code_object_offset, slice_size, nobits)

   @ brief Given a code object and an ELF virtual address, map the ELF virtual
   address to a code object offset.

   Also, determine if the ELF virtual address
   maps to an offset in a data region that is defined by the ELF file, but that
   does not occupy bytes in the ELF file. This is typically true of offsets that
   that refer to runtime or heap allocated memory. For ELF files with defined
   sections, these data regions are referred to as NOBITS or .bss sections.

   For bits regions: the size in bytes, starting from the provided virtual
   address up to either the end of the segment, or the start of a NOBITS region.
   In this case, slice size represents the number of contiguous readable
   addresses following the provided address.

   @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
   successfully.

   @retval ::AMD_COMGR_STATUS_ERROR The provided code object has an invalid
   header due to a mismatch in magic, class, data, version, abi, type, or
   machine.

   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``data`` is not of
   kind AMD_COMGR_DATA_KIND_EXECUTABLE or invalid, or that the provided `elf_virtual_address` is not within the ranges covered by the object's
   load-type program headers.

   Args:
       data (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           The data object to be inspected for the given ELF virtual
           address. This should be of kind AMD_COMGR_DATA_KIND_EXECUTABLE.

       elf_virtual_address (:py:obj:`~.int`) -- *IN*:
           The address used to calculate the code object
           offset.

       code_object_offset (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           The code object offset returned to the caller
           based on the given ELF virtual address.

       slice_size (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           For nobits regions: the size in bytes, starting from
           the provided virtual address up to the end of the segment. In this case, the
           slice size represents the number of contiguous unreadable addresses following
           the provided address.

       nobits (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Set to true if the code object offset points to a location
           in a data region that does not occupy bytes in the ELF file, as described
           above.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_map_elf_virtual_address_to_code_object_offset(amd_comgr_data_t data, uint64_t elf_virtual_address, uint64_t * code_object_offset, uint64_t * slice_size, _Bool * nobits)


.. py:function:: amd_comgr_hotswap_rewrite(input, source_isa_name, target_isa_name, output)

   Rewrite a code object from one ISA to another.

   Rewrites GPU instructions in the ELF code object so that it can execute
   on a different target ISA. This includes both same-family stepping
   patches (e.g. B0 to A0) and cross-family transpilation.
   The input ELF is not modified; a new data object is created and returned.

   A successful call means COMGR produced a valid output code object, not
   necessarily that the output bytes differ from the input. If the
   source/target ISA pair selects no enabled transformation, the output is a
   copy of the input.

   Currently supported transformations:
     - GFX1250 B0 to A0

   Additional source/target ISA pairs may be added in future releases.
   Unsupported ``source_isa_name`` / ``target_isa_name`` combinations return
   ``AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT.``

   @retval ::AMD_COMGR_STATUS_SUCCESS Patching completed successfully.
   @retval ::AMD_COMGR_STATUS_ERROR An internal error occurred.
   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``input`` is an invalid
     data object, is not of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE,`` does not
     contain data bytes, or ``source_isa_name`` or ``target_isa_name`` is NULL,
     or the source/target isa name combination is not supported.
   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to allocate
     the output data object.

   Args:
       input (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE``
           containing the input ELF code object bytes.

       source_isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name
           the code object was compiled for. The isa name is defined as the Code
           Object Target Identification string, described at
           https://llvm.org/docs/AMDGPUUsage.html:py:obj:`~.code`-object-target-identification

       target_isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name
           of the target GPU.

       output (:py:obj:`~.amd_comgr_data_s`/:py:obj:`~.object`) -- *OUT*:
           A handle to a data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` containing the rewritten ELF. The caller
           must release this handle using ``amd_comgr_release_data`` when done.
           ``output`` is not modified on failure.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_hotswap_rewrite(amd_comgr_data_t input, const char * source_isa_name, const char * target_isa_name, amd_comgr_data_t * output)


.. py:class:: amd_comgr_hotswap_rewrite_flag_s

   Bases: :py:obj:`enum.IntEnum`


   HotSwap rewrite option flags.
       


   .. py:attribute:: AMD_COMGR_HOTSWAP_REWRITE_FLAG_NONE
      :type:  int


   .. py:attribute:: AMD_COMGR_HOTSWAP_REWRITE_FLAG_ENTRY_TRAMPOLINES
      :type:  int


   .. py:attribute:: AMD_COMGR_HOTSWAP_REWRITE_FLAG_STRICT_MODE
      :type:  int


.. py:data:: amd_comgr_hotswap_rewrite_flag_t

.. py:class:: amd_comgr_hotswap_rewrite_options_s(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Options for ``amd_comgr_hotswap_rewrite_with_options.``
       


   .. py:attribute:: size
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


.. py:data:: amd_comgr_hotswap_rewrite_options_t

.. py:function:: amd_comgr_hotswap_rewrite_with_options(input, source_isa_name, target_isa_name, rewrite_options, output)

   Rewrite a code object from one ISA to another with explicit options.

   Rewrites GPU instructions in the ELF code object so that it can execute
   on a different target ISA. This includes both same-family stepping
   patches (e.g. B0 to A0) and cross-family transpilation.
   The input ELF is not modified; a new data object is created and returned.

   A successful call means COMGR produced a valid output code object, not
   necessarily that the output bytes differ from the input. If the
   source/target ISA pair and rewrite options select no enabled transformation,
   the output is a copy of the input.

   Currently supported transformations:
     - GFX1250 B0 to A0
     - GFX125x entry trampolines when requested by ``rewrite_options``
     - GFX1250 B0 strict-mode mask workarounds when requested by
       ``rewrite_options``

   Additional source/target ISA pairs may be added in future releases.
   Unsupported ``source_isa_name`` / ``target_isa_name`` combinations return
   ``AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT.``

   @retval ::AMD_COMGR_STATUS_SUCCESS Patching completed successfully.
   @retval ::AMD_COMGR_STATUS_ERROR An internal error occurred.
   @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ``input`` is an invalid
     data object, is not of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE,`` does not
     contain data bytes, ``source_isa_name,`` ``target_isa_name,`` `rewrite_options`, or ``output`` is NULL, the source/target isa name
     combination is not supported, the options structure is too small, or
     unsupported option flags are set.
   @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to allocate
     the output data object.

   Args:
       input (:py:obj:`~.amd_comgr_data_s`) -- *IN*:
           A data object of kind ``AMD_COMGR_DATA_KIND_EXECUTABLE``
           containing the input ELF code object bytes.

       source_isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name
           the code object was compiled for. The isa name is defined as the Code
           Object Target Identification string, described at
           https://llvm.org/docs/AMDGPUUsage.html:py:obj:`~.code`-object-target-identification

       target_isa_name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           A null terminated string that is the isa name
           of the target GPU.

       rewrite_options (:py:obj:`~.amd_comgr_hotswap_rewrite_options_s`/:py:obj:`~.object`) -- *IN*:
           Options controlling optional rewrite behavior.
           Must not be NULL. Unknown flag bits return
           ``AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT.``

       output (:py:obj:`~.amd_comgr_data_s`/:py:obj:`~.object`) -- *OUT*:
           A handle to a data object of kind `AMD_COMGR_DATA_KIND_EXECUTABLE` containing the rewritten ELF. The caller
           must release this handle using ``amd_comgr_release_data`` when done.
           ``output`` is not modified on failure.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.amd_comgr_status_s`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       amd_comgr_status_t amd_comgr_hotswap_rewrite_with_options(amd_comgr_data_t input, const char * source_isa_name, const char * target_isa_name, const amd_comgr_hotswap_rewrite_options_t * rewrite_options, amd_comgr_data_t * output)


