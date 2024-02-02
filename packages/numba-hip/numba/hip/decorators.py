from warnings import warn
from numba.core import types, config, sigutils
from numba.core.errors import DeprecationError, NumbaInvalidConfigWarning
from numba.hip.compiler import declare_device_function
from numba.hip.dispatcher import HIPDispatcher
# from numba.hip.simulator.kernel import FakeHIPKernel # TODO support simulator

_msg_deprecated_signature_arg = ("Deprecated keyword argument `{0}`. "
                                 "Signatures should be passed as the first "
                                 "positional argument.")


def jit(func_or_sig=None, device=False, inline=False, link=[], debug=None,
        opt=True, lineinfo=False, cache=False, **kws):
    """
    JIT compile a Python function for AMD GPUs.

    :param func_or_sig: A function to JIT compile, or *signatures* of a
       function to compile. If a function is supplied, then a
       :class:`Dispatcher <numba.hip.dispatcher.HIPDispatcher>` is returned.
       Otherwise, ``func_or_sig`` may be a signature or a list of signatures,
       and a function is returned. The returned function accepts another
       function, which it will compile and then return a :class:`Dispatcher
       <numba.hip.dispatcher.HIPDispatcher>`. See :ref:`jit-decorator` for
       more information about passing signatures.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param link: A list of files containing LLVM IR/BC or HIP C++ source to link
       with the function
    :type link: list
    :param debug: If True, check for exceptions thrown when executing the
       kernel. Since this degrades performance, this should only be used for
       debugging purposes. If set to True, then ``opt`` should be set to False.
       Defaults to False.  (The default value can be overridden by setting
       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param fastmath: When True, enables fastmath optimizations.
    :param max_registers: Request that the kernel is limited to using at most
       this number of registers per thread. The limit may not be respected if
       the ABI requires a greater number of registers than that requested.
       Useful for increasing occupancy.
    :param opt: (UNSUPPORTED) Set optimization level to 3 (``True``) or 0 (``False``).
    :type opt: bool
    :param lineinfo: (UNSUPPORTED) If ``True``, generate a line mapping between source code and
       assembly code. This enables inspection of the source code in AMD GPU
       profiling tools and correlation with program counter sampling.
    :type lineinfo: bool
    :param cache: If True, enables the file-based cache for this function.
    :type cache: bool
    """
    # TODO enable unsupported option

    # TODO implement simulator
    # if link and config.ENABLE_CUDASIM:
    #     raise NotImplementedError('Cannot link PTX in the simulator')

    if kws.get('boundscheck'):
        raise NotImplementedError("bounds checking is not supported for HIP")

    if kws.get('argtypes') is not None:
        msg = _msg_deprecated_signature_arg.format('argtypes')
        raise DeprecationError(msg)
    if kws.get('restype') is not None:
        msg = _msg_deprecated_signature_arg.format('restype')
        raise DeprecationError(msg)
    if kws.get('bind') is not None:
        msg = _msg_deprecated_signature_arg.format('bind')
        raise DeprecationError(msg)

    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug # TODO add HIP option
    fastmath = kws.get('fastmath', False)
    extensions = kws.get('extensions', [])

    if debug and opt:
        msg = ("debug=True with opt=True (the default) "
               "is not supported by HIP. This may result in a crash"
               " - set debug=False or opt=False.") # TODO check if that's also the case for HIP
        warn(NumbaInvalidConfigWarning(msg))

    if debug and lineinfo:
        msg = ("debug and lineinfo are mutually exclusive. Use debug to get "
               "full debug info (this disables some optimizations), or "
               "lineinfo for line info only with code generation unaffected.")
        warn(NumbaInvalidConfigWarning(msg)) # TODO check if that's also the case for HIP

    if device and kws.get('link'):
        raise ValueError("link keyword invalid for device function")

    if sigutils.is_signature(func_or_sig):
        signatures = [func_or_sig]
        specialized = True
    elif isinstance(func_or_sig, list):
        signatures = func_or_sig
        specialized = False
    else:
        signatures = None

    if signatures is not None:
        # TODO support HIP simulator
        # if config.ENABLE_CUDASIM:
        #     def jitwrapper(func):
        #         return FakeCUDAKernel(func, device=device, fastmath=fastmath)
        #     return jitwrapper

        def _jit(func):
            targetoptions = kws.copy()
            targetoptions['debug'] = debug
            targetoptions['lineinfo'] = lineinfo
            targetoptions['link'] = link
            targetoptions['opt'] = opt
            targetoptions['fastmath'] = fastmath
            targetoptions['device'] = device
            targetoptions['extensions'] = extensions

            disp = HIPDispatcher(func, targetoptions=targetoptions)

            if cache:
                disp.enable_caching()

            for sig in signatures:
                argtypes, restype = sigutils.normalize_signature(sig)

                if restype and not device and restype != types.void:
                    raise TypeError("HIP kernel must have void return type.")

                if device:
                    from numba.core import typeinfer
                    with typeinfer.register_dispatcher(disp):
                        disp.compile_device(argtypes, restype)
                else:
                    disp.compile(argtypes)

            disp._specialized = specialized
            disp.disable_compile()

            return disp

        return _jit
    else:
        if func_or_sig is None:
            if config.ENABLE_CUDASIM:
                raise NotImplementedError()
                # TODO support HIP simulator
                # def autojitwrapper(func):
                #     return FakeHIPKernel(func, device=device,
                #                           fastmath=fastmath)
            else:
                def autojitwrapper(func):
                    return jit(func, device=device, debug=debug, opt=opt,
                               lineinfo=lineinfo, link=link, cache=cache, **kws)

            return autojitwrapper
        # func_or_sig is a function
        else:
            if config.ENABLE_CUDASIM:
                raise NotImplementedError()
                # TODO support HIP simulator
                # return FakeHIPKernel(func_or_sig, device=device,
                #                       fastmath=fastmath)
            else:
                targetoptions = kws.copy()
                targetoptions['debug'] = debug
                targetoptions['lineinfo'] = lineinfo
                targetoptions['opt'] = opt
                targetoptions['link'] = link
                targetoptions['fastmath'] = fastmath
                targetoptions['device'] = device
                targetoptions['extensions'] = extensions
                disp = HIPDispatcher(func_or_sig, targetoptions=targetoptions)

                if cache:
                    disp.enable_caching()

                return disp


def declare_device(name, sig):
    """
    Declare the signature of a foreign function. Returns a descriptor that can
    be used to call the function from a Python kernel.

    :param name: The name of the foreign function.
    :type name: str
    :param sig: The Numba signature of the function.
    """
    argtypes, restype = sigutils.normalize_signature(sig)
    if restype is None:
        msg = 'Return type must be provided for device declarations'
        raise TypeError(msg)

    return declare_device_function(name, restype, argtypes)
