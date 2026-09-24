rocm.bindings.hiprtc
====================

.. py:module:: rocm.bindings.hiprtc


Attributes
----------

.. autoapisummary::

   rocm.bindings.hiprtc.hiprtcLinkState
   rocm.bindings.hiprtc.hiprtcProgram


Classes
-------

.. autoapisummary::

   rocm.bindings.hiprtc.hiprtcResult
   rocm.bindings.hiprtc.ihiprtcLinkState


Functions
---------

.. autoapisummary::

   rocm.bindings.hiprtc.has_symbol
   rocm.bindings.hiprtc.hiprtcGetErrorString
   rocm.bindings.hiprtc.hiprtcVersion
   rocm.bindings.hiprtc.hiprtcAddNameExpression
   rocm.bindings.hiprtc.hiprtcCompileProgram
   rocm.bindings.hiprtc.hiprtcCreateProgram
   rocm.bindings.hiprtc.hiprtcDestroyProgram
   rocm.bindings.hiprtc.hiprtcGetLoweredName
   rocm.bindings.hiprtc.hiprtcGetProgramLog
   rocm.bindings.hiprtc.hiprtcGetProgramLogSize
   rocm.bindings.hiprtc.hiprtcGetCode
   rocm.bindings.hiprtc.hiprtcGetCodeSize
   rocm.bindings.hiprtc.hiprtcGetBitcode
   rocm.bindings.hiprtc.hiprtcGetBitcodeSize
   rocm.bindings.hiprtc.hiprtcLinkCreate
   rocm.bindings.hiprtc.hiprtcLinkAddFile
   rocm.bindings.hiprtc.hiprtcLinkAddData
   rocm.bindings.hiprtc.hiprtcLinkComplete
   rocm.bindings.hiprtc.hiprtcLinkDestroy


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: hiprtcResult

   Bases: :py:obj:`enum.IntEnum`


   *

   hiprtc error code


   .. py:attribute:: HIPRTC_SUCCESS
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_OUT_OF_MEMORY
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_INVALID_INPUT
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_INVALID_PROGRAM
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_INVALID_OPTION
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_COMPILATION
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_INTERNAL_ERROR
      :type:  int


   .. py:attribute:: HIPRTC_ERROR_LINKING
      :type:  int


.. py:class:: ihiprtcLinkState(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hiprtcLinkState

.. py:function:: hiprtcGetErrorString(result)

   Returns text string message to explain the error which occurred

   Warning:
       In HIP, this function returns the name of the error,
       if the hiprtc result is defined, it will return "Invalid HIPRTC error code"

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       result (:py:obj:`~.hiprtcResult`) -- *IN*:
           code to convert to string.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`:
               Always returns `~.hiprtcResult.HIPRTC_SUCCESS`.
       * :py:obj:`~.bytes`: const char pointer to the NULL-terminated error string

   .. rubric:: C signature

   .. code-block:: c

       const char * hiprtcGetErrorString(hiprtcResult result)


.. py:function:: hiprtcVersion()

   Sets the parameters as major and minor version.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`, :py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~.int`:
               HIP Runtime Compilation major version.
       * :py:obj:`~.int`:
               HIP Runtime Compilation minor version.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcVersion(int * major, int * minor)


.. py:data:: hiprtcProgram

.. py:function:: hiprtcAddNameExpression(prog, name_expression)

   Adds the given name exprssion to the runtime compilation program.

   If const char pointer is NULL, it will return :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       name_expression (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           const char pointer to the name expression.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char * name_expression)


.. py:function:: hiprtcCompileProgram(prog, numOptions, options)

   Compiles the given runtime compilation program.

   If the compiler failed to build the runtime compilation program,
   it will return :py:obj:`~.HIPRTC_ERROR_COMPILATION`.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       numOptions (:py:obj:`~.int`) -- *IN*:
           number of compiler options.

       options (:py:obj:`~.rocm.bindings.util.types.ListOfBytes`/:py:obj:`~.object`) -- *IN*:
           compiler options as const array of strins.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char *const * options)


.. py:function:: hiprtcCreateProgram(src, name, numHeaders, headers, includeNames)

   Creates an instance of hiprtcProgram with the given input parameters,
   and sets the output hiprtcProgram prog with it.

   Any invalide input parameter, it will return :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`
   or :py:obj:`~.HIPRTC_ERROR_INVALID_PROGRAM`.

   If failed to create the program, it will return :py:obj:`~.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE`.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       src (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           const char pointer to the program source.

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           const char pointer to the program name.

       numHeaders (:py:obj:`~.int`) -- *IN*:
           number of headers.

       headers (:py:obj:`~.rocm.bindings.util.types.ListOfBytes`/:py:obj:`~.object`) -- *IN*:
           array of strings pointing to headers.

       includeNames (:py:obj:`~.rocm.bindings.util.types.ListOfBytes`/:py:obj:`~.object`) -- *IN*:
           array of strings pointing to names included in program source.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~._hiprtcProgram`:
               runtime compilation program instance.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcCreateProgram(hiprtcProgram * prog, const char * src, const char * name, int numHeaders, const char *const * headers, const char *const * includeNames)


.. py:function:: hiprtcDestroyProgram(prog)

   Destroys an instance of given hiprtcProgram.

   If prog is NULL, it will return :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcDestroyProgram(hiprtcProgram * prog)


.. py:function:: hiprtcGetLoweredName(prog, name_expression)

   Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input
   parameters, and sets the output lowered_name with it.

   If any invalide nullptr input parameters, it will return :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`

   If name_expression is not found, it will return :py:obj:`~.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID`

   If failed to get lowered_name from the program, it will return :py:obj:`~.HIPRTC_ERROR_COMPILATION`.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       name_expression (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           const char pointer to the name expression.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               const char array to the lowered (mangled) name.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog, const char * name_expression, const char ** lowered_name)


.. py:function:: hiprtcGetProgramLog(prog, log)

   Gets the log generated by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       log (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *OUT*:
           memory pointer to the generated log.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char * log)


.. py:function:: hiprtcGetProgramLogSize(prog)

   Gets the size of log generated by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~.int`:
               size of generated log.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t * logSizeRet)


.. py:function:: hiprtcGetCode(prog, code)

   Gets the pointer of compilation binary by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       code (:py:obj:`~.rocm.bindings.util.types.NDBuffer`/:py:obj:`~.object`) -- *OUT*:
           char pointer to binary.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetCode(hiprtcProgram prog, char * code)


.. py:function:: hiprtcGetCodeSize(prog)

   Gets the size of compilation binary by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~.int`:
               the size of binary.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t * codeSizeRet)


.. py:function:: hiprtcGetBitcode(prog, bitcode)

   Gets the pointer of compiled bitcode by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

       bitcode (:py:obj:`~.rocm.bindings.util.types.NDBuffer`/:py:obj:`~.object`) -- *OUT*:
           char pointer to bitcode.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: HIPRTC_SUCCESS

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char * bitcode)


.. py:function:: hiprtcGetBitcodeSize(prog)

   Gets the size of compiled bitcode by the runtime compilation program instance.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       prog (:py:obj:`~._hiprtcProgram`/:py:obj:`~.object`) -- *IN*:
           runtime compilation program instance.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`
       * :py:obj:`~.int`:
               the size of bitcode.

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t * bitcode_size)


.. py:function:: hiprtcLinkCreate(num_options, option_ptr, option_vals_pptr)

   Creates the link instance via hiprtc APIs.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       num_options (:py:obj:`~.int`) -- *IN*:
           Number of options

       option_ptr (:py:obj:`~.rocm.bindings._hiprtc_helpers.HiprtcLinkCreate_option_ptr`/:py:obj:`~.object`) -- *IN*:
           Array of options

       option_vals_pptr (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`, :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`, :py:obj:`~.HIPRTC_ERROR_INVALID_OPTION`
       * :py:obj:`~.ihiprtcLinkState`:
               hiprtc link state created upon success

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcLinkCreate(unsigned int num_options, hipJitOption * option_ptr, void ** option_vals_pptr, hiprtcLinkState * hip_link_state_ptr)


.. py:function:: hiprtcLinkAddFile(hip_link_state, input_type, file_path, num_options, options_ptr, option_values)

   Adds a file with bit code to be linked with options

   If input values are invalid, it will

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       hip_link_state (:py:obj:`~.ihiprtcLinkState`/:py:obj:`~.object`) -- *IN*:
           hiprtc link state

       input_type (:py:obj:`~.hipJitInputType`) -- *IN*:
           Type of the input data or bitcode

       file_path (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Path to the input file where bitcode is present

       num_options (:py:obj:`~.int`) -- *IN*:
           Size of the options

       options_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Array of options applied to this input

       option_values (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: One of:
               - py:obj:`~.HIPRTC_SUCCESS`
               - py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state, hipJitInputType input_type, const char * file_path, unsigned int num_options, hipJitOption * options_ptr, void ** option_values)


.. py:function:: hiprtcLinkAddData(hip_link_state, input_type, image, image_size, name, num_options, options_ptr, option_values)

   Completes the linking of the given program.

   If adding the file fails, it will

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       hip_link_state (:py:obj:`~.ihiprtcLinkState`/:py:obj:`~.object`) -- *IN*:
           hiprtc link state

       input_type (:py:obj:`~.hipJitInputType`) -- *IN*:
           Type of the input data or bitcode

       image (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data which is null terminated

       image_size (:py:obj:`~.int`) -- *IN*:
           Size of the input data

       name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           Optional name for this input

       num_options (:py:obj:`~.int`) -- *IN*:
           Size of the options

       options_ptr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Array of options applied to this input

       option_values (:py:obj:`~.rocm.bindings.util.types.ListOfPointer`/:py:obj:`~.object`) -- *IN*:
           Array of option values cast to void*

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: One of:
               - py:obj:`~.HIPRTC_SUCCESS`, :py:obj:`~.HIPRTC_ERROR_INVALID_INPUT`
               - py:obj:`~.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state, hipJitInputType input_type, void * image, size_t image_size, const char * name, unsigned int num_options, hipJitOption * options_ptr, void ** option_values)


.. py:function:: hiprtcLinkComplete(hip_link_state)

   Completes the linking of the given program.

   If adding the data fails, it will

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       hip_link_state (:py:obj:`~.ihiprtcLinkState`/:py:obj:`~.object`) -- *IN*:
           hiprtc link state

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: One of:
               - py:obj:`~.HIPRTC_SUCCESS`
               - py:obj:`~.HIPRTC_ERROR_LINKING`
       * :py:obj:`~.rocm.bindings.util.types.NDBuffer`/:py:obj:`~.object`:
               Upon success, points to the output binary
       * :py:obj:`~.int`:
               Size of the binary is stored (optional)

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state, void ** bin_out, size_t * size_out)


.. py:function:: hiprtcLinkDestroy(hip_link_state)

   Deletes the link instance via hiprtc APIs.

   See:
       :py:obj:`~.hiprtcResult`

   Args:
       hip_link_state (:py:obj:`~.ihiprtcLinkState`/:py:obj:`~.object`) -- *IN*:
           link state instance

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprtcResult`: py:obj:`~.HIPRTC_SUCCESS`

   .. rubric:: C signature

   .. code-block:: c

       hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state)


