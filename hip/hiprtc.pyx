# AMD_COPYRIGHT
import cython
import ctypes
import enum
class hiprtcResult(enum.IntEnum):
    HIPRTC_SUCCESS = chiprtc.HIPRTC_SUCCESS
    HIPRTC_ERROR_OUT_OF_MEMORY = chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    HIPRTC_ERROR_INVALID_INPUT = chiprtc.HIPRTC_ERROR_INVALID_INPUT
    HIPRTC_ERROR_INVALID_PROGRAM = chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    HIPRTC_ERROR_INVALID_OPTION = chiprtc.HIPRTC_ERROR_INVALID_OPTION
    HIPRTC_ERROR_COMPILATION = chiprtc.HIPRTC_ERROR_COMPILATION
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    HIPRTC_ERROR_INTERNAL_ERROR = chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
    HIPRTC_ERROR_LINKING = chiprtc.HIPRTC_ERROR_LINKING
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class hiprtcJIT_option(enum.IntEnum):
    HIPRTC_JIT_MAX_REGISTERS = chiprtc.HIPRTC_JIT_MAX_REGISTERS
    HIPRTC_JIT_THREADS_PER_BLOCK = chiprtc.HIPRTC_JIT_THREADS_PER_BLOCK
    HIPRTC_JIT_WALL_TIME = chiprtc.HIPRTC_JIT_WALL_TIME
    HIPRTC_JIT_INFO_LOG_BUFFER = chiprtc.HIPRTC_JIT_INFO_LOG_BUFFER
    HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = chiprtc.HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    HIPRTC_JIT_ERROR_LOG_BUFFER = chiprtc.HIPRTC_JIT_ERROR_LOG_BUFFER
    HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = chiprtc.HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    HIPRTC_JIT_OPTIMIZATION_LEVEL = chiprtc.HIPRTC_JIT_OPTIMIZATION_LEVEL
    HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = chiprtc.HIPRTC_JIT_TARGET_FROM_HIPCONTEXT
    HIPRTC_JIT_TARGET = chiprtc.HIPRTC_JIT_TARGET
    HIPRTC_JIT_FALLBACK_STRATEGY = chiprtc.HIPRTC_JIT_FALLBACK_STRATEGY
    HIPRTC_JIT_GENERATE_DEBUG_INFO = chiprtc.HIPRTC_JIT_GENERATE_DEBUG_INFO
    HIPRTC_JIT_LOG_VERBOSE = chiprtc.HIPRTC_JIT_LOG_VERBOSE
    HIPRTC_JIT_GENERATE_LINE_INFO = chiprtc.HIPRTC_JIT_GENERATE_LINE_INFO
    HIPRTC_JIT_CACHE_MODE = chiprtc.HIPRTC_JIT_CACHE_MODE
    HIPRTC_JIT_NEW_SM3X_OPT = chiprtc.HIPRTC_JIT_NEW_SM3X_OPT
    HIPRTC_JIT_FAST_COMPILE = chiprtc.HIPRTC_JIT_FAST_COMPILE
    HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_NAMES
    HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS
    HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_COUNT
    HIPRTC_JIT_LTO = chiprtc.HIPRTC_JIT_LTO
    HIPRTC_JIT_FTZ = chiprtc.HIPRTC_JIT_FTZ
    HIPRTC_JIT_PREC_DIV = chiprtc.HIPRTC_JIT_PREC_DIV
    HIPRTC_JIT_PREC_SQRT = chiprtc.HIPRTC_JIT_PREC_SQRT
    HIPRTC_JIT_FMA = chiprtc.HIPRTC_JIT_FMA
    HIPRTC_JIT_NUM_OPTIONS = chiprtc.HIPRTC_JIT_NUM_OPTIONS
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class hiprtcJITInputType(enum.IntEnum):
    HIPRTC_JIT_INPUT_CUBIN = chiprtc.HIPRTC_JIT_INPUT_CUBIN
    HIPRTC_JIT_INPUT_PTX = chiprtc.HIPRTC_JIT_INPUT_PTX
    HIPRTC_JIT_INPUT_FATBINARY = chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    HIPRTC_JIT_INPUT_OBJECT = chiprtc.HIPRTC_JIT_INPUT_OBJECT
    HIPRTC_JIT_INPUT_LIBRARY = chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    HIPRTC_JIT_INPUT_NVVM = chiprtc.HIPRTC_JIT_INPUT_NVVM
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    HIPRTC_JIT_INPUT_LLVM_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_BITCODE
    HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE
    HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
    HIPRTC_JIT_NUM_INPUT_TYPES = chiprtc.HIPRTC_JIT_NUM_INPUT_TYPES
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


cdef class ihiprtcLinkState:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef ihiprtcLinkState from_ptr(chiprtc.ihiprtcLinkState* ptr, bint owner=False):
        """Factory function to create ``ihiprtcLinkState`` objects from
        given ``chiprtc.ihiprtcLinkState`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihiprtcLinkState wrapper = ihiprtcLinkState.__new__(ihiprtcLinkState)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef ihiprtcLinkState from_pyobj(object pyobj):
        """Derives a ihiprtcLinkState from a Python object.

        Derives a ihiprtcLinkState from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``ihiprtcLinkState`` reference, this method
        returns it directly. No new ``ihiprtcLinkState`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ihiprtcLinkState``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of ihiprtcLinkState!
        """
        cdef ihiprtcLinkState wrapper = ihiprtcLinkState.__new__(ihiprtcLinkState)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,ihiprtcLinkState):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chiprtc.ihiprtcLinkState*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chiprtc.ihiprtcLinkState*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chiprtc.ihiprtcLinkState*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chiprtc.ihiprtcLinkState*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<ihiprtcLinkState object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
    @staticmethod
    def PROPERTIES():
        return []

    def __contains__(self,item):
        properties = self.PROPERTIES()
        return item in properties

    def __getitem__(self,item):
        properties = self.PROPERTIES()
        if isinstance(item,int):
            if item < 0 or item >= len(properties):
                raise IndexError()
            return getattr(self,properties[item])
        raise ValueError("'item' type must be 'int'")


hiprtcLinkState = ihiprtcLinkState

@cython.embedsignature(True)
def hiprtcGetErrorString(object result):
    """@brief Returns text string message to explain the error which occurred
    @param [in] result  code to convert to string.
    @return  const char pointer to the NULL-terminated error string
    @warning In HIP, this function returns the name of the error,
    if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
    @see hiprtcResult
    """
    if not isinstance(result,hiprtcResult):
        raise TypeError("argument 'result' must be of type 'hiprtcResult'")
    cdef const char * _hiprtcGetErrorString__retval = chiprtc.hiprtcGetErrorString(result.value)    # fully specified
    return (_hiprtcGetErrorString__retval,)


@cython.embedsignature(True)
def hiprtcVersion():
    """@brief Sets the parameters as major and minor version.
    @param [out] major  HIP Runtime Compilation major version.
    @param [out] minor  HIP Runtime Compilation minor version.
    """
    cdef int major
    cdef int minor
    _hiprtcVersion__retval = hiprtcResult(chiprtc.hiprtcVersion(&major,&minor))    # fully specified
    return (_hiprtcVersion__retval,major,minor)


cdef class _hiprtcProgram:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef _hiprtcProgram from_ptr(chiprtc._hiprtcProgram* ptr, bint owner=False):
        """Factory function to create ``_hiprtcProgram`` objects from
        given ``chiprtc._hiprtcProgram`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef _hiprtcProgram wrapper = _hiprtcProgram.__new__(_hiprtcProgram)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef _hiprtcProgram from_pyobj(object pyobj):
        """Derives a _hiprtcProgram from a Python object.

        Derives a _hiprtcProgram from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``_hiprtcProgram`` reference, this method
        returns it directly. No new ``_hiprtcProgram`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``_hiprtcProgram``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of _hiprtcProgram!
        """
        cdef _hiprtcProgram wrapper = _hiprtcProgram.__new__(_hiprtcProgram)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,_hiprtcProgram):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chiprtc._hiprtcProgram*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chiprtc._hiprtcProgram*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chiprtc._hiprtcProgram*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chiprtc._hiprtcProgram*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<_hiprtcProgram object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
    @staticmethod
    def PROPERTIES():
        return []

    def __contains__(self,item):
        properties = self.PROPERTIES()
        return item in properties

    def __getitem__(self,item):
        properties = self.PROPERTIES()
        if isinstance(item,int):
            if item < 0 or item >= len(properties):
                raise IndexError()
            return getattr(self,properties[item])
        raise ValueError("'item' type must be 'int'")


hiprtcProgram = _hiprtcProgram

@cython.embedsignature(True)
def hiprtcAddNameExpression(object prog, const char * name_expression):
    """@brief Adds the given name exprssion to the runtime compilation program.
    @param [in] prog  runtime compilation program instance.
    @param [in] name_expression  const char pointer to the name expression.
    @return  HIPRTC_SUCCESS
    If const char pointer is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    _hiprtcAddNameExpression__retval = hiprtcResult(chiprtc.hiprtcAddNameExpression(
        _hiprtcProgram.from_pyobj(prog)._ptr,name_expression))    # fully specified
    return (_hiprtcAddNameExpression__retval,)


@cython.embedsignature(True)
def hiprtcCompileProgram(object prog, int numOptions, object options):
    """@brief Compiles the given runtime compilation program.
    @param [in] prog  runtime compilation program instance.
    @param [in] numOptions  number of compiler options.
    @param [in] options  compiler options as const array of strins.
    @return HIPRTC_SUCCESS
    If the compiler failed to build the runtime compilation program,
    it will return HIPRTC_ERROR_COMPILATION.
    @see hiprtcResult
    """
    _hiprtcCompileProgram__retval = hiprtcResult(chiprtc.hiprtcCompileProgram(
        _hiprtcProgram.from_pyobj(prog)._ptr,numOptions,
        <const char **>hip._util.types.ListOfBytes.from_pyobj(options)._ptr))    # fully specified
    return (_hiprtcCompileProgram__retval,)


@cython.embedsignature(True)
def hiprtcCreateProgram(const char * src, const char * name, int numHeaders, object headers, object includeNames):
    """@brief Creates an instance of hiprtcProgram with the given input parameters,
    and sets the output hiprtcProgram prog with it.
    @param [in, out] prog  runtime compilation program instance.
    @param [in] src  const char pointer to the program source.
    @param [in] name  const char pointer to the program name.
    @param [in] numHeaders  number of headers.
    @param [in] headers  array of strings pointing to headers.
    @param [in] includeNames  array of strings pointing to names included in program source.
    @return HIPRTC_SUCCESS
    Any invalide input parameter, it will return HIPRTC_ERROR_INVALID_INPUT
    or HIPRTC_ERROR_INVALID_PROGRAM.
    If failed to create the program, it will return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE.
    @see hiprtcResult
    """
    prog = _hiprtcProgram.from_ptr(NULL)
    _hiprtcCreateProgram__retval = hiprtcResult(chiprtc.hiprtcCreateProgram(&prog._ptr,src,name,numHeaders,
        <const char **>hip._util.types.ListOfBytes.from_pyobj(headers)._ptr,
        <const char **>hip._util.types.ListOfBytes.from_pyobj(includeNames)._ptr))    # fully specified
    return (_hiprtcCreateProgram__retval,prog)


@cython.embedsignature(True)
def hiprtcDestroyProgram(object prog):
    """@brief Destroys an instance of given hiprtcProgram.
    @param [in] prog  runtime compilation program instance.
    @return HIPRTC_SUCCESS
    If prog is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    _hiprtcDestroyProgram__retval = hiprtcResult(chiprtc.hiprtcDestroyProgram(
        <chiprtc.hiprtcProgram*>hip._util.types.DataHandle.from_pyobj(prog)._ptr))    # fully specified
    return (_hiprtcDestroyProgram__retval,)


@cython.embedsignature(True)
def hiprtcGetLoweredName(object prog, const char * name_expression):
    """@brief Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input parameters,
    and sets the output lowered_name with it.
    @param [in] prog  runtime compilation program instance.
    @param [in] name_expression  const char pointer to the name expression.
    @param [in, out] lowered_name  const char array to the lowered (mangled) name.
    @return HIPRTC_SUCCESS
    If any invalide nullptr input parameters, it will return HIPRTC_ERROR_INVALID_INPUT
    If name_expression is not found, it will return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    If failed to get lowered_name from the program, it will return HIPRTC_ERROR_COMPILATION.
    @see hiprtcResult
    """
    cdef const char * lowered_name
    _hiprtcGetLoweredName__retval = hiprtcResult(chiprtc.hiprtcGetLoweredName(
        _hiprtcProgram.from_pyobj(prog)._ptr,name_expression,&lowered_name))    # fully specified
    return (_hiprtcGetLoweredName__retval,lowered_name)


@cython.embedsignature(True)
def hiprtcGetProgramLog(object prog, object log):
    """@brief Gets the log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] log  memory pointer to the generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    _hiprtcGetProgramLog__retval = hiprtcResult(chiprtc.hiprtcGetProgramLog(
        _hiprtcProgram.from_pyobj(prog)._ptr,
        <char *>hip._util.types.DataHandle.from_pyobj(log)._ptr))    # fully specified
    return (_hiprtcGetProgramLog__retval,)


@cython.embedsignature(True)
def hiprtcGetProgramLogSize(object prog):
    """@brief Gets the size of log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] logSizeRet  size of generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    cdef unsigned long logSizeRet
    _hiprtcGetProgramLogSize__retval = hiprtcResult(chiprtc.hiprtcGetProgramLogSize(
        _hiprtcProgram.from_pyobj(prog)._ptr,&logSizeRet))    # fully specified
    return (_hiprtcGetProgramLogSize__retval,logSizeRet)


@cython.embedsignature(True)
def hiprtcGetCode(object prog, object code):
    """@brief Gets the pointer of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    _hiprtcGetCode__retval = hiprtcResult(chiprtc.hiprtcGetCode(
        _hiprtcProgram.from_pyobj(prog)._ptr,
        <char *>hip._util.types.DataHandle.from_pyobj(code)._ptr))    # fully specified
    return (_hiprtcGetCode__retval,)


@cython.embedsignature(True)
def hiprtcGetCodeSize(object prog):
    """@brief Gets the size of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    cdef unsigned long codeSizeRet
    _hiprtcGetCodeSize__retval = hiprtcResult(chiprtc.hiprtcGetCodeSize(
        _hiprtcProgram.from_pyobj(prog)._ptr,&codeSizeRet))    # fully specified
    return (_hiprtcGetCodeSize__retval,codeSizeRet)


@cython.embedsignature(True)
def hiprtcGetBitcode(object prog, object bitcode):
    """@brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    _hiprtcGetBitcode__retval = hiprtcResult(chiprtc.hiprtcGetBitcode(
        _hiprtcProgram.from_pyobj(prog)._ptr,
        <char *>hip._util.types.DataHandle.from_pyobj(bitcode)._ptr))    # fully specified
    return (_hiprtcGetBitcode__retval,)


@cython.embedsignature(True)
def hiprtcGetBitcodeSize(object prog):
    """@brief Gets the size of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    cdef unsigned long bitcode_size
    _hiprtcGetBitcodeSize__retval = hiprtcResult(chiprtc.hiprtcGetBitcodeSize(
        _hiprtcProgram.from_pyobj(prog)._ptr,&bitcode_size))    # fully specified
    return (_hiprtcGetBitcodeSize__retval,bitcode_size)


@cython.embedsignature(True)
def hiprtcLinkCreate(unsigned int num_options, object option_ptr, object option_vals_pptr):
    """@brief Creates the link instance via hiprtc APIs.
    @param [in] hip_jit_options
    @param [out] hiprtc link state instance
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    hip_link_state_ptr = ihiprtcLinkState.from_ptr(NULL)
    _hiprtcLinkCreate__retval = hiprtcResult(chiprtc.hiprtcLinkCreate(num_options,
        <chiprtc.hiprtcJIT_option *>hip._util.types.DataHandle.from_pyobj(option_ptr)._ptr,
        <void **>hip._util.types.DataHandle.from_pyobj(option_vals_pptr)._ptr,&hip_link_state_ptr._ptr))    # fully specified
    return (_hiprtcLinkCreate__retval,hip_link_state_ptr)


@cython.embedsignature(True)
def hiprtcLinkAddFile(object hip_link_state, object input_type, const char * file_path, unsigned int num_options, object options_ptr, object option_values):
    """@brief Adds a file with bit code to be linked with options
    @param [in] hiprtc link state, jit input type, file path,
    option reated parameters.
    @param [out] None.
    @return HIPRTC_SUCCESS
    If input values are invalid, it will
    @return HIPRTC_ERROR_INVALID_INPUT
    @see hiprtcResult
    """
    if not isinstance(input_type,hiprtcJITInputType):
        raise TypeError("argument 'input_type' must be of type 'hiprtcJITInputType'")
    _hiprtcLinkAddFile__retval = hiprtcResult(chiprtc.hiprtcLinkAddFile(
        ihiprtcLinkState.from_pyobj(hip_link_state)._ptr,input_type.value,file_path,num_options,
        <chiprtc.hiprtcJIT_option *>hip._util.types.DataHandle.from_pyobj(options_ptr)._ptr,
        <void **>hip._util.types.DataHandle.from_pyobj(option_values)._ptr))    # fully specified
    return (_hiprtcLinkAddFile__retval,)


@cython.embedsignature(True)
def hiprtcLinkAddData(object hip_link_state, object input_type, object image, unsigned long image_size, const char * name, unsigned int num_options, object options_ptr, object option_values):
    """@brief Completes the linking of the given program.
    @param [in] hiprtc link state, jit input type, image_ptr ,
    option reated parameters.
    @param [out] None.
    @return HIPRTC_SUCCESS
    If adding the file fails, it will
    @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    @see hiprtcResult
    """
    if not isinstance(input_type,hiprtcJITInputType):
        raise TypeError("argument 'input_type' must be of type 'hiprtcJITInputType'")
    _hiprtcLinkAddData__retval = hiprtcResult(chiprtc.hiprtcLinkAddData(
        ihiprtcLinkState.from_pyobj(hip_link_state)._ptr,input_type.value,
        <void *>hip._util.types.DataHandle.from_pyobj(image)._ptr,image_size,name,num_options,
        <chiprtc.hiprtcJIT_option *>hip._util.types.DataHandle.from_pyobj(options_ptr)._ptr,
        <void **>hip._util.types.DataHandle.from_pyobj(option_values)._ptr))    # fully specified
    return (_hiprtcLinkAddData__retval,)


@cython.embedsignature(True)
def hiprtcLinkComplete(object hip_link_state):
    """@brief Completes the linking of the given program.
    @param [in] hiprtc link state instance
    @param [out] linked_binary, linked_binary_size.
    @return HIPRTC_SUCCESS
    If adding the data fails, it will
    @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    @see hiprtcResult
    """
    bin_out = hip._util.types.DataHandle.from_ptr(NULL)
    cdef unsigned long size_out
    _hiprtcLinkComplete__retval = hiprtcResult(chiprtc.hiprtcLinkComplete(
        ihiprtcLinkState.from_pyobj(hip_link_state)._ptr,
        <void **>&bin_out._ptr,&size_out))    # fully specified
    return (_hiprtcLinkComplete__retval,bin_out,size_out)


@cython.embedsignature(True)
def hiprtcLinkDestroy(object hip_link_state):
    """@brief Deletes the link instance via hiprtc APIs.
    @param [in] hiprtc link state instance
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    If linking fails, it will
    @return HIPRTC_ERROR_LINKING
    @see hiprtcResult
    """
    _hiprtcLinkDestroy__retval = hiprtcResult(chiprtc.hiprtcLinkDestroy(
        ihiprtcLinkState.from_pyobj(hip_link_state)._ptr))    # fully specified
    return (_hiprtcLinkDestroy__retval,)
