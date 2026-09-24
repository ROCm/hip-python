rocm.comgr.comgr
================

.. py:module:: rocm.comgr.comgr

.. autoapi-nested-parse::

   Higher level interfaces that simplify the use of AMD COMGR.

   Attributes:
       HIPRTC_RUNTIME_HEADER (`str`):
           The content of the ``hipRTC`` / ``hiprtc_runtime.h`` header file, read on
           first access from the installed ROCm's ``hiprtc-builtins`` library. See
           :py:mod:`rocm.comgr.hiprtc_header`, which explains why the installed
           copy is the only correct one.
           Take a look at https://github.com/ROCm/clr for more details on how this
           file is generated.



Classes
-------

.. autoapisummary::

   rocm.comgr.comgr.Symbol
   rocm.comgr.comgr.Data
   rocm.comgr.comgr.DataSet
   rocm.comgr.comgr.Action


Functions
---------

.. autoapisummary::

   rocm.comgr.comgr.to_bytes
   rocm.comgr.comgr.to_str
   rocm.comgr.comgr.comgr_check
   rocm.comgr.comgr.metadata_string_get_bytes
   rocm.comgr.comgr.metadata_map_get_keys
   rocm.comgr.comgr.parse_metadata
   rocm.comgr.comgr.parse_data_metadata
   rocm.comgr.comgr.parse_code_obj_metadata
   rocm.comgr.comgr.parse_code_obj_kernel_names
   rocm.comgr.comgr.get_isa_names
   rocm.comgr.comgr.get_isa_metadata
   rocm.comgr.comgr.get_isa_metadata_all
   rocm.comgr.comgr.parse_data_symbols
   rocm.comgr.comgr.parse_code_symbols
   rocm.comgr.comgr.disassemble_program
   rocm.comgr.comgr.disassemble_code_obj_function
   rocm.comgr.comgr.dump_metadata_yaml
   rocm.comgr.comgr.disassemble_amdhsa_code_obj_v6_kernel
   rocm.comgr.comgr.compile_hip_to_bc
   rocm.comgr.comgr.compile_bc_to_hsa
   rocm.comgr.comgr.compile_hip_to_hsa
   rocm.comgr.comgr.compile_bc
   rocm.comgr.comgr.compile_hsa
   rocm.comgr.comgr.disassemble_via_action_deprecated


Module Contents
---------------

.. py:function:: to_bytes(obj)

.. py:function:: to_str(obj)

.. py:function:: comgr_check(call_result)

   Check AMD COMGR call status and return other result tuple entries.


.. py:function:: metadata_string_get_bytes(metadata_string: rocm.bindings.amd_comgr.amd_comgr_metadata_node_s)

   Get the text associated with a string metadata node as `bytes`.

   Args:
       metadata_str (`~.amd_comgr_metadata_node_s`):
           A metadata node that represents a string.


.. py:function:: metadata_map_get_keys(metadata_map: rocm.bindings.amd_comgr.amd_comgr_metadata_node_s)

   Get the keys of a AMD COMGR metadata map as `list`.

   Args:
       metadata_map (`~.amd_comgr_metadata_node_s`):
           A metadata node that represents a map.


.. py:function:: parse_metadata(metadata: rocm.bindings.amd_comgr.amd_comgr_metadata_node_s, level: int = 0)

   Parse metadata node and return a nest of `dict`, `list`, and `str`.

   Args:
       metadata_map (`~.amd_comgr_metadata_node_s`):
           A metadata node that represents a map.
       level (int, optional):
           Not used yet. Useful for debugging.


.. py:function:: parse_data_metadata(data: rocm.bindings.amd_comgr.amd_comgr_data_s)

   Parse metadata of a `amd_comgr_data_s` object.


.. py:function:: parse_code_obj_metadata(code_obj, code_obj_size, kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE)

   Parse metadata of a code object, e.g., one generated via HIPRTC.

   Args:
       code_obj:
           Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
           e.g. an implementor of the Python buffer protocol such as `bytes`.
       code_obj_size (`int`):
           Length of the code.
       kind (`~.amd_comgr_data_kind_s`, optional):
           Kind of the code object in terms of AMD COMGR kinds, e.g.
           `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
           which is the default.


.. py:function:: parse_code_obj_kernel_names(code, code_size, kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE)

   Return the names of kernels in the code object.

   Results are returned in order of appearance.

   Args:
       code:
           Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
           e.g. an implementor of the Python buffer protocol such as `bytes`.
       code_size (`int`):
           Length of the code.
       kind (`~.amd_comgr_data_kind_s`, optional):
           Kind of the code object in terms of AMD COMGR kinds, e.g.
           `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
           which is the default.


.. py:function:: get_isa_names(decode: bool = True) -> list

   Return list of ISA names supported by this COMGR version

   Return ISA names supported by this COMGR version as `list` of `str` or
   `bytes`.

   Args:
       decode (`bool`, optional):
           If the names should be decoded to a Python `str`.
   Returns:
       `list`:
           List of ISA names, either as Python `str` (``decode=True``) or
           `bytes`.


.. py:function:: get_isa_metadata(isa_name)

   Parse metadata for a specific ISA.

   Args:
       isa_name (`bytes` or `str`):
           The ISA name
   See:
       get_isa_names


.. py:function:: get_isa_metadata_all()

   Return metadata for all ISAs supported by this COMGR version as dict.


.. py:class:: Symbol

   Represents a code symbol in a code object.

   Members result from `~.amd_comgr_symbol_get_info` supplied with the
   following ``attribute`` parameters:

   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME_LENGTH`:
       The length of the symbol name in bytes. Does not include the NUL
       terminator. The type of this attribute is uint64_t.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME`:
       The name of the symbol. The type of this attribute is character array
       with the length equal to the value of the
       AMD_COMGR_SYMBOL_INFO_NAME_LENGTH attribute plus 1 for a NUL
       terminator.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_TYPE`:
       The kind of the symbol. The type of this attribute is
       amd_comgr_symbol_type_t.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_SIZE`:
       Size of the variable. The value of this attribute is undefined if the
       symbol is not a variable. The type of this attribute is uint64_t.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED`:
       Indicates whether the symbol is undefined. The type of this attribute
       is bool.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_VALUE`:
       The value of the symbol. The type of this attribute is uint64_t.
   `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_LAST`:
       Marker for last valid symbol info.

   An object's member ``self.type`` can have the following values:

   ``UNKNOWN``:
       The symbol's type is unknown.
   ``NOTYPE``:
       The symbol's type is not specified.
   ``OBJECT``:
       The symbol is associated with a data object, such as a variable, an
       array, and so on.
   ``FUNC``:
       The symbol is associated with a function or other executable code.
   ``SECTION``:
       The symbol is associated with a section. Symbol table entries of this
       type exist primarily for relocation.
   ``FILE``:
       Conventionally, the symbol's name gives the name of the source file
       associated with the object file.
   ``COMMON``:
       The symbol labels an uninitialized common block.
   ``AMDGPU_HSA_KERNEL``:
       The symbol is associated with an AMDGPU Code Object V2 kernel function.


   .. py:attribute:: type
      :value: None



   .. py:attribute:: name
      :value: None



   .. py:attribute:: size
      :value: -1



   .. py:attribute:: is_undefined
      :value: -1



   .. py:attribute:: value
      :value: -1



.. py:function:: parse_data_symbols(data: rocm.bindings.amd_comgr.amd_comgr_data_s)

   Parse all symbols of a data object and return as `dict`.


.. py:function:: parse_code_symbols(code, code_size, kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE)

   Parse metadata of a code object, e.g., one generated via HIPRTC.

   Args:
       code:
           Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
           e.g. an implementor of the Python buffer protocol such as `bytes`.
       code_size (`int`):
           Length of the code.
       kind (`~.amd_comgr_data_kind_s`, optional):
           Kind of the code object in terms of AMD COMGR kinds, e.g.
           `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
           which is the default.


.. py:class:: Data(name: str, kind_str: str, source_buffer=None)

   .. py:method:: kind_str_to_enum(kind_str: str)
      :staticmethod:


      Look up an `~.amd_comgr_data_kind_s` value by name.

      Accepts either the short form (``"SOURCE"``,
      ``"FATBIN"``, ...) or the full enum name
      (``"AMD_COMGR_DATA_KIND_SOURCE"``); the
      ``AMD_COMGR_DATA_KIND_`` prefix is added if not already
      present. Case-insensitive.

      The canonical list of valid values lives on the autogenerated
      cy* enum class `~.amd_comgr_data_kind_s` — strip the
      ``AMD_COMGR_DATA_KIND_`` prefix from each member name to get
      the short-name form. Use :py:meth:`valid_kinds` for a
      runtime-derived list.

      Raises:
          AttributeError: if the (normalised) name is not a member
              of the cy* enum (e.g. a key removed upstream in a
              newer ROCm release than the binding was built
              against).



   .. py:method:: valid_kinds() -> list[str]
      :staticmethod:


      Return the short-name keys accepted by
      :py:meth:`kind_str_to_enum` — derived at runtime from
      the cy* enum, so always in sync with the installed wheel.



   .. py:attribute:: kind_str


   .. py:method:: get_data_name() -> str

      Returns the data object's name as `str`.



   .. py:method:: get_data_len()

      Get the size of the managed data as Python 'int'.



   .. py:method:: get_data_bytes() -> bytes

      Copy the managed data into a new buffer and return it as Python
      'bytes'.

      Note:
          This routine should only be used if this is a result data object,
          e.g. obtained as result from an action. If this is a source data
          object, you can also access ``self.source_bytes`` for a copy of
          the original source data buffer.



   .. py:method:: get()


.. py:class:: DataSet(*datas)

   .. py:attribute:: datas
      :value: []



   .. py:method:: add_data(data: Data)


   .. py:method:: count_data(kind_str: str)


   .. py:method:: get_data(kind_str: str, index: int) -> Data


   .. py:method:: get()


.. py:class:: Action(action_kind_str: str, isa_name=None, lang_str=None, options=None, logging: bool = False)

   .. py:method:: action_kind_str_to_enum(action_kind_str: str)
      :staticmethod:


      Look up an `~.amd_comgr_action_kind_s` value by name.

      Accepts either the short form (``"COMPILE_SOURCE_TO_BC"``,
      ``"LINK_BC_TO_BC"``, ...) or the full enum name
      (``"AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC"``); the
      ``AMD_COMGR_ACTION_`` prefix is added if not already
      present. Case-insensitive.

      The canonical list of valid values lives on the autogenerated
      cy* enum class `~.amd_comgr_action_kind_s` — strip the
      ``AMD_COMGR_ACTION_`` prefix from each member name to get the
      short-name form. Use :py:meth:`valid_action_kinds` for a
      runtime-derived list.

      Notes:
          A few keys that earlier ROCm releases exposed have been
          removed upstream (passing them will raise
          `AttributeError`):

          * ``ADD_DEVICE_LIBRARIES`` — removed in ROCm 6.4;
            replaced by ``COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC``
            (fuses the device-libs link into the compile step).
          * ``OPTIMIZE_BC_TO_BC`` — removed in an earlier release.
          * ``COMPILE_SOURCE_TO_FATBIN`` — removed in an earlier
            release; the fat-binary path is now reached via
            ``COMPILE_SOURCE_TO_EXECUTABLE`` /
            ``COMPILE_SOURCE_TO_RELOCATABLE``.

      Raises:
          AttributeError: if the (normalised) name is not a member
              of the cy* enum (e.g. a key removed upstream in a
              newer ROCm release than the binding was built
              against).



   .. py:method:: valid_action_kinds() -> list[str]
      :staticmethod:


      Return the short-name keys accepted by
      :py:meth:`action_kind_str_to_enum` — derived at runtime from
      the cy* enum, so always in sync with the installed wheel.



   .. py:method:: lang_str_to_enum(lang_str: str)
      :staticmethod:


      Look up an `~.amd_comgr_language_s` value by name.

      Accepts either the short form (``"HIP"``, ``"LLVM_IR"``, ...)
      or the full enum name (``"AMD_COMGR_LANGUAGE_HIP"``); the
      ``AMD_COMGR_LANGUAGE_`` prefix is added if not already
      present. Case-insensitive.

      The canonical list of valid values lives on the autogenerated
      cy* enum class `~.amd_comgr_language_s` — strip the
      ``AMD_COMGR_LANGUAGE_`` prefix from each member name to get
      the short-name form. Use :py:meth:`valid_languages` for a
      runtime-derived list.

      Notes:
          ``HC`` (AMD Heterogeneous C++) was supported by earlier
          ROCm releases but has been retired — passing it will
          raise `AttributeError`. Use ``HIP`` instead.

      Raises:
          AttributeError: if the (normalised) name is not a member
              of the cy* enum (e.g. a key removed upstream in a
              newer ROCm release than the binding was built
              against).



   .. py:method:: valid_languages() -> list[str]
      :staticmethod:


      Return the short-name keys accepted by
      :py:meth:`lang_str_to_enum` — derived at runtime from the
      cy* enum, so always in sync with the installed wheel.



   .. py:attribute:: result_data_set


   .. py:method:: set_isa_name(isa_name)

      Sets the action info object's isa_name.

      Args:
          isa_name (`str` or Python buffer such as `bytes`):
              ISA name supported by this version of AMD COMGR, e.g.
              ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
              See `~.get_isa_names`, `~.get_isa_metadata_all` for more
              information.
      Note:
          Input will be null-terminated if it is not already.



   .. py:method:: set_language(lang_str: str)

      Set the language of the input data, e.g. "HIP".

      See:
          `lang_str_to_enum`.



   .. py:method:: set_options(options)

      Set options to supply to the action runner.

      Args:
          options (`list` or `tuple` of Python buffer such as `bytes`):
              Options to supply to the action runner.
      Note:
          Input will be null-terminated if it is not already.



   .. py:method:: get_num_options()


   .. py:method:: get_option(index: int) -> bytes


   .. py:method:: set_logging(logging: bool)

      Enable logging.



   .. py:method:: do_action(input_data_set: DataSet, check=True) -> rocm.bindings.amd_comgr.amd_comgr_status_s

      Run the action for the given data_set.



   .. py:method:: get()


.. py:function:: disassemble_program(isa_name, program, append_address_annotation=False, read_memory_cb=_default_read_memory_cb, append_instruction_cb=_default_append_instruction_cb, append_address_annotation_cb=_default_append_address_annotation_cb)

   Disassemble an AMD GPU machine code program.

   Note:
       DISASSEMBLE_* Actions will soon be deprecated;
       see: https://github.com/<internal-amd-org>/llvm-project/pull/2677

   Args:
       program:
           A block of machine code that represents a sequence of instructions.


.. py:function:: disassemble_code_obj_function(code_obj, isa_name, func_name=None, append_address_annotation=False)

   Disassembles a kernel or device function stored in a code object.

   Identifies the size and location of the function symbol in the code object
   and disassembles it.

   Args:
       code_obj:
           An object that can be converted to bytes.
       func_name (`str` or `None`):
           The name of the function lookup or `None`.
           If `None` is specified, the first found function
           is disassembled.


.. py:function:: dump_metadata_yaml(metadata_dict)

   Note:
       AMD HSA kernel metadata does not
       have list of lists or dict of dicts.
       We see only dict of lists, dict of values
       and list of dicts.


.. py:function:: disassemble_amdhsa_code_obj_v6_kernel(code_obj, isa_name, kernel_name=None, append_address_annotation=False, raw=False)

   Disassembles a kernel stored in an AMD GPU code object v6.

   Identifies the size and location of the function symbol in the code object
   and disassembles it.

   Args:
       code_obj:
           An object that can be converted to bytes.
       func_name (`str` or `None`, ooptional):
           The name of the function lookup or `None`.
           If `None` is specified, the first found kernel
           is disassembled.
       raw (`bool`, optional):
           Just return raw instructions, do not prepend and
           append code object v6 specific ELF directions
           and metadata. Defaults to `False`.



.. py:function:: compile_hip_to_bc(source, isa_name, hip_version_tuple, extra_opts=[], default_opts=['-fgpu-rdc', '-O3', '-mcumode', '-std=c++14', '-nogpuinc', '-Wno-gnu-line-marker', '-Wno-missing-prototypes'], prepend_hiprtc_runtime_header=False, logging=False, action_kind='COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC')

   Compiles a HIP C++ source to LLVM BC.

   Args:
       source (`str` or Python buffer such as `bytes`):
           The input as bytes or str.
       isa_name (`str` or Python buffer such as `bytes`):
           ISA name supported by this version of AMD COMGR, e.g.
           ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
           See `~.get_isa_names`, `~.get_isa_metadata_all` for more
           information.
       hip_version_tuple (`tuple[int]`):
           Integer triple like ``(6,0,32830)`` that indicates a HIP version.
       extra_opts (`list` of `str` or Python buffer such as `bytes`):
           Extra options that are appended to the default options; see
           argument ``default_opts``.
           You would typically supply additional options via this value but
           can also use it overrule some or all of the options specified
           in default_opts. Defaults to `[]`.
       default_opts (`list` of `str` or Python buffer such as `bytes`):
           Default options that are typically not changed.
           Defaults to `["-fgpu-rdc", "-O3", "-mcumode", "-std=c++14",
           "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
       prepend_hiprtc_runtime_header (`bool`, optional):
           Prepend the hiprtc runtime header to the source code.
           Defaults to `False`.
       logging (`bool`):
           Enable logging. Defaults to ``False``.
       action_kind (`str`):
           The compile action kind. Defaults to
           ``"COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC"``
           Other supported option is ``"COMPILE_SOURCE_TO_BC"``.

   Returns:
       `tuple`:
           A `tuple` of size 1 with the following components (in that order):
           1. `bytes`: The compilation result, an LLVM BC file.
           2. `str` or `None`: The log output if logging was specified.
           3. `str` or `None`: The diagnostics output if diagnostics were
               enabled via options.

   Raises:
       `RuntimeError`:
           If one of the compile fails. Enable logging to get more
           detailed error reports.

   Note:
       String arguments are always encoded as `utf-8`.
   Note:
       Default option `-fgpu-rdc` keeps `__device__` functions in the bitcode
       file.
   See:
       `~.get_isa_names`, `~.get_isa_metadata_all`

   Note:
       This implementation is based on what AMD COMGR logs to screen when
       compiling HIP code to BC via hipRTC while the environment variables
       ``AMD_COMGR_REDIRECT_LOGS="stderr"`` and
       ``AMD_COMGR_EMIT_VERBOSE_LOGS=1`` are active:

       ```text
       ActionKind: AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC
       IsaName: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
       Options: "-O3" "-mcumode" "--hip-version=6.0.32830"
       "-DHIP_VERSION_MAJOR=6" "-DHIP_VERSION_MINOR=0"
       "-DHIP_VERSION_PATCH=32830" "-D__HIPCC_RTC__" "-include"
       "hiprtc_runtime.h" "-std=c++14" "-nogpuinc" "-Wno-gnu-line-marker"
       "-Wno-missing-prototypes" "--offload-arch=gfx90a:sramecc+:xnack-"
       "-fgpu-rdc"
       Path:
           Language: AMD_COMGR_LANGUAGE_HIP
       Compilation Args: [...]
       Driver Job Args: [...]
           ReturnStatus: AMD_COMGR_STATUS_SUCCESS
       ```


.. py:function:: compile_bc_to_hsa(source, isa_name, bc_kind='BC', extra_opts=[], logging=False)

   Translate LLVM IR/BC to HSA.

   Args:
       source (`str` or Python buffer such as `bytes`):
           The input as bytes or str.
       isa_name (`str` or Python buffer such as `bytes`):
           ISA name supported by this version of AMD COMGR, e.g.
           ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
           See `~.get_isa_names`, `~.get_isa_metadata_all` for more
           information.
       bc_kind (`str`, optional):
           Either "BC" or "BC_BUNDLE". Defaults to "BC".
       extra_opts (`list` of `str` or `bytes`-like, optional):
           Extra options that are appended to the default options.
           You would typically supply additional options via this value but
           can also use it overrule some or all of the options specified
           in default_opts. Defaults to `[]`.
       logging (`bool`, optional):
           Enable logging. Defaults to ``False``.

   Returns:
       `tuple`:
           A `tuple` of size 1 with the following components (in that order):

           1. `bytes`: The compilation result, an AMD GPU HSA assembly source
               file.
           2. `str` or `None`: The log output if logging was specified.
           3. `str` or `None`: The diagnostics output if diagnostics were
               enabled via options.

   Raises:
       `RuntimeError`:
           If one of the compile fails. Enable logging to get more
           detailed error reports.

   Note:
       String arguments are always encoded as `utf-8`.
   See:
       `~.get_isa_names`, `~.get_isa_metadata_all`
       ```


.. py:function:: compile_hip_to_hsa(source, isa_name, hip_version_tuple, extra_opts=[], default_opts=['-S', '-O3', '-mcumode', '-std=c++14', '-nogpuinc', '-Wno-gnu-line-marker', '-Wno-missing-prototypes'], prepend_hiprtc_runtime_header=False, logging=False)

   Compiles a HIP C++ source to HSA.

   Args:
       source (`str` or Python buffer such as `bytes`):
           The input as bytes or str.
       isa_name (`str` or Python buffer such as `bytes`):
           ISA name supported by this version of AMD COMGR, e.g.
           ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
           See `~.get_isa_names`, `~.get_isa_metadata_all` for more
           information.
       hip_version_tuple (`tuple[int]`):
           Integer triple like ``(6,0,32830)`` that indicates a HIP version.
       extra_opts (`list` of `str` or Python buffer such as `bytes`):
           Extra options that are appended to the default options; see
           argument ``default_opts``.
           You would typically supply additional options via this value but
           can also use it overrule some or all of the options specified
           in default_opts. Defaults to `[]`.
       default_opts (`list` of `str` or Python buffer such as `bytes`):
           Default options that are typically not changed.
           Defaults to `["-S", "-O3", "-mcumode", "-std=c++14",
           "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
       prepend_hiprtc_runtime_header (`bool`, optional)
           Prepend the hiprtc runtime header to the source code.
           Defaults to `False`.
       logging (bool):
           Enable logging. Defaults to ``False``.

   Returns:
       `tuple`:
           A `tuple` of size 3 with the following components (in that order):
           1. `bytes`: The compilation result, an AMD GPU HSA assembly source
              file.
           2. `str` or `None`: The log output if logging was specified.
           3. `str` or `None`: The diagnostics output if diagnostics were
               enabled via options.

   Raises:
       `RuntimeError`:
           If one of the compile fails. Enable logging to get more
           detailed error reports.

   Note:
       String arguments are always encoded as `utf-8`.
   See:
       `~.get_isa_names`, `~.get_isa_metadata_all`
       ```


.. py:function:: compile_bc(ir_or_bc, isa_name, bc_kind='BC', extra_opts=[], default_opts=['-O3'], logging=False)

   Compile LLVM BC/IR to AMD GPU code object.

   Args:
       ir_or_bc (`str` or Python buffer such as `bytes`):
           The input as bytes or str.
       isa_name (`str` or Python buffer such as `bytes`):
           ISA name supported by this version of AMD COMGR, e.g.
           ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
           See `~.get_isa_names`, `~.get_isa_metadata_all` for more
           information.
       bc_kind (`str`, optional):
           Either "BC" or "BC_BUNDLE". Defaults to "BC".
       extra_opts (`list` of `str` or Python buffer such as `bytes`):
           Extra options that are appended to the default options; see
           argument ``default_opts``.
           You would typically supply additional options via this value but
           can also use it overrule some or all of the options specified
           in default_opts. Defaults to `[]`.
       default_opts (`list` of `str` or Python buffer such as `bytes`):
           Default options that are typically not changed.
           Defaults to `["-S", "-O3", "-mcumode", "-std=c++14",
           "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
       logging (bool):
           Enable logging. Defaults to ``False``.

   Returns:
       `tuple`:
           A `tuple` of size 3 with the following components (in that order):
           1. `bytes`: The compilation result, an AMD GPU object in ELF
              format.
           2. `str` or `None`: The log output if logging was specified.
           3. `str` or `None`: The diagnostics output if diagnostics were
              enabled via options.

   Raises:
       `RuntimeError`:
           If one of the compile fails. Enable logging to get more
           detailed error reports.

   Note:
       We deduced the order of actions and their parametrization from the
       AMD COMGR log output when running hiprtcLinkComplete with
       BC input. We decided to skip the LINK_BC_TO_BC step.


.. py:function:: compile_hsa(hsa, isa_name, extra_opts=[], default_opts=[], logging=False)

   Compile AMD HSA assembly to AMD GPU code object.

   Args:
       source (`str` or Python buffer such as `bytes`):
           The input as bytes or str.
       isa_name (`str` or Python buffer such as `bytes`):
           ISA name supported by this version of AMD COMGR, e.g.
           ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
           See `~.get_isa_names`, `~.get_isa_metadata_all` for more
           information.
       extra_opts (`list` of `str` or Python buffer such as `bytes`):
           Extra options that are appended to the default options; see
           argument ``default_opts``.
           You would typically supply additional options via this value but
           can also use it overrule some or all of the options specified
           in default_opts. Defaults to `[]`.
       default_opts (`list` of `str` or Python buffer such as `bytes`):
           Default options that are typically not changed.
           Defaults to `[]`.
       logging (bool):
           Enable logging. Defaults to ``False``.

   Returns:
       `tuple`:
           A `tuple` of size 3 with the following components (in that order):
           1. `bytes`: The compilation result, an AMD GPU object in ELF
              format.
           2. `str` or `None`: The log output if logging was specified.
           3. `str` or `None`: The diagnostics output if diagnostics were
              enabled via options.

   Raises:
       `RuntimeError`:
           If one of the compile fails. Enable logging to get more
           detailed error reports.

   Note:
       We deduced the order of actions and their parametrization from the
       AMD COMGR log output when running hiprtcLinkComplete with
       BC input. We decided to skip the LINK_BC_TO_BC step.


.. py:function:: disassemble_via_action_deprecated(code_obj, isa_name, logging=False, action_kind='DISASSEMBLE_EXECUTABLE_TO_SOURCE')

   Disassemble an AMD GPU executable/relocatable.

   Warning:
       DISASSEMBLE_* Actions will soon be deprecated;
       see: https://github.com/<internal-amd-org>/llvm-project/pull/2677



