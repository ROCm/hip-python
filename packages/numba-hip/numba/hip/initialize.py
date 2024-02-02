def initialize_all():
    """Register the ROC extension.
    """
    # todo FIXME
    # Import models to register them with the data model manager
    #import numba.cuda.models  # noqa: F401
    #
    #from numba.cuda.decorators import jit
    #from numba.cuda.dispatcher import CUDADispatcher
    from numba.core.target_extension import (target_registry,
                                            dispatcher_registry,
                                            jit_registry)

    # hip_target = target_registry["hip"]
    # jit_registry[hip_target] = jit
    # dispatcher_registry[roc_target] = CUDADispatcher