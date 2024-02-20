def initialize_all():
    """Register the HIP extension."""
    # todo FIXME
    # Import models to register them with the data model manager
    import numba.hip.typing_lowering.models  # noqa: F401

    #
    from numba.hip.decorators import jit
    from numba.hip.dispatcher import HIPDispatcher
    from numba.core.target_extension import (
        target_registry,
        dispatcher_registry,
        jit_registry,
        GPU,
    )

    class HIP(GPU):
        pass

    target_registry["hip"] = HIP

    hip_target = target_registry["hip"]
    jit_registry[hip_target] = jit
    dispatcher_registry[hip_target] = HIPDispatcher
