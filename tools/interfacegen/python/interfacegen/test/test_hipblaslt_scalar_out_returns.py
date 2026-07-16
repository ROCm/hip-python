# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Tests for the scalar callee-allocated OUT overrides in
``support.recipes.rocm.hipblaslt``.

hipBLASLt reuses hipblas's LAPACK-oriented ``ptr_rank``, which leaves a
plain ``int*`` / ``size_t*`` at the default rank 1. Without an override a
documented ``@param[out]`` scalar only refines to plain ``OUT`` (a
``ListOf*`` caller argument), and ``hipblasLtGetVersion``'s *untagged*
``version`` isn't even seen as OUT. The ``hipblaslt._SCALAR_OUT_PARMS``
table pins these to ``OUT_CALLEE_ALLOCATED`` + rank 0 so the generated
wrapper *returns* the value.

These tests use synthetic single-function headers (no dependency on a real
hipblaslt.h) so they run wherever pytest can.
"""

from interfacegen import cython, treefactory
from interfacegen.cparser import CParser
from interfacegen.support.recipes import rocm
from interfacegen.support.recipes.control import ParmIntent


def _build(header_text: str):
    parser = CParser("input.h", unsaved_files=[("input.h", header_text)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython, translation_unit=parser.translation_unit,
    )


def _bind(root, recipe):
    """Attach the recipe's ``ptr_rank`` / intent callbacks to every Parm,
    mirroring ``interfacegen.cython._backend.initialize_nodes``.

    Required because ``documented_param_intent`` consults ``parm.ptr_rank``
    (via ``_is_callee_allocated_out_shape``) to decide whether a documented
    ``[out]`` scalar refines to the callee-allocated flavor.
    """
    for n in root.walk(postorder=True):
        if isinstance(n, cython.Parm):
            setattr(n, "ptr_rank", recipe.ptr_rank)
            setattr(n, "ptr_intent", recipe.ptr_parm_intent)


def _parm(root, fname, pname):
    for n in root.walk(postorder=False):
        if isinstance(n, cython.Function) and n.name == fname:
            for p in n.parms:
                if p.name == pname:
                    return p
    raise KeyError(f"{fname}({pname})")


def _assert_scalar_return(recipe, p):
    """A scalar callee-allocated return is OUT_CALLEE_ALLOCATED + rank 0."""
    intent = recipe.ptr_parm_intent(p)
    assert intent == ParmIntent.OUT_CALLEE_ALLOCATED
    assert intent.direction == ParmIntent.OUT
    assert recipe.ptr_rank(p) == 0


def test_untagged_version_becomes_scalar_return():
    """``hipblasLtGetVersion(handle, int* version)`` has NO ``@param`` tag
    upstream, so the generic chain leaves it at the INOUT fallback (a
    caller ``ListOf*`` argument). The ``_SCALAR_OUT_PARMS`` override forces
    the callee-allocated scalar return.
    """
    root = _build("""
        typedef void* hipblasLtHandle_t;
        int hipblasLtGetVersion(hipblasLtHandle_t handle, int* version);
    """)
    _bind(root, rocm.hipblaslt)
    _assert_scalar_return(rocm.hipblaslt, _parm(root, "hipblasLtGetVersion", "version"))


def test_documented_size_written_becomes_scalar_return():
    """``sizeWritten`` is documented ``@param[out]`` but hipblas's rank rule
    would leave it at rank 1 (plain ``OUT`` → ``ListOfUnsignedLong`` arg).
    The override upgrades it to a returned scalar. The caller-sized ``buf``
    attribute buffer is deliberately NOT upgraded.
    """
    root = _build("""
        typedef void* hipblasLtMatmulDesc_t;
        typedef int hipblasLtMatmulDescAttributes_t;
        /**
         *  @param[in]  buf         caller attribute buffer.
         *  @param[in]  sizeInBytes buffer capacity in bytes.
         *  @param[out] sizeWritten bytes written / needed.
         */
        int hipblasLtMatmulDescGetAttribute(
            hipblasLtMatmulDesc_t matmulDesc,
            hipblasLtMatmulDescAttributes_t attr,
            void* buf,
            unsigned long sizeInBytes,
            unsigned long* sizeWritten);
    """)
    _bind(root, rocm.hipblaslt)
    _assert_scalar_return(
        rocm.hipblaslt, _parm(root, "hipblasLtMatmulDescGetAttribute", "sizeWritten")
    )
    # The attribute buffer stays a caller-allocated argument (not a return).
    buf = _parm(root, "hipblasLtMatmulDescGetAttribute", "buf")
    assert rocm.hipblaslt.ptr_parm_intent(buf) != ParmIntent.OUT_CALLEE_ALLOCATED


def test_sm_count_target_and_nan_call_id_are_scalar_returns():
    root = _build("""
        typedef void* hipblasLtHandle_t;
        /** @param[out] smCountTarget the SM-count target. */
        int hipblasLtGetSmCountTarget(hipblasLtHandle_t handle, int* smCountTarget);
        /** @param[out] first_nan_call_id the call id of the first NaN. */
        int hipblasLtCheckNumericsDrain(hipblasLtHandle_t handle,
                                        unsigned int* first_nan_call_id);
    """)
    _bind(root, rocm.hipblaslt)
    _assert_scalar_return(
        rocm.hipblaslt, _parm(root, "hipblasLtGetSmCountTarget", "smCountTarget")
    )
    _assert_scalar_return(
        rocm.hipblaslt,
        _parm(root, "hipblasLtCheckNumericsDrain", "first_nan_call_id"),
    )


def test_return_algo_count_scalar_but_results_array_caller_allocated():
    """The scalar ``returnAlgoCount`` becomes a return, while the sibling
    caller-sized ``heuristicResultsArray[]`` MUST stay caller-allocated
    (rank-1 OUT) — the override targets only the scalar count.
    """
    root = _build("""
        typedef void* hipblasLtHandle_t;
        typedef struct { int state; } hipblasLtMatmulHeuristicResult_t;
        /**
         *  @param[in]  requestedAlgoCount     array capacity.
         *  @param[out] heuristicResultsArray  caller-allocated results.
         *  @param[out] returnAlgoCount        number of results written.
         */
        int hipblasLtMatmulAlgoGetHeuristic(
            hipblasLtHandle_t handle,
            int requestedAlgoCount,
            hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
            int* returnAlgoCount);
    """)
    _bind(root, rocm.hipblaslt)
    _assert_scalar_return(
        rocm.hipblaslt, _parm(root, "hipblasLtMatmulAlgoGetHeuristic", "returnAlgoCount")
    )
    arr = _parm(root, "hipblasLtMatmulAlgoGetHeuristic", "heuristicResultsArray")
    assert rocm.hipblaslt.ptr_parm_intent(arr) == ParmIntent.OUT
    assert rocm.hipblaslt.ptr_rank(arr) == 1


def test_handle_creator_still_returns_via_hipblas_tail():
    """The override delegates its tail to hipblas, so the ``void** handle``
    creator heuristic (``hipblasLtCreate``) still yields a returned handle.
    """
    root = _build("""
        typedef void* hipblasLtHandle_t;
        int hipblasLtCreate(hipblasLtHandle_t* handle);
    """)
    _bind(root, rocm.hipblaslt)
    p = _parm(root, "hipblasLtCreate", "handle")
    assert rocm.hipblaslt.ptr_parm_intent(p) == ParmIntent.OUT_CALLEE_ALLOCATED
    assert rocm.hipblaslt.ptr_rank(p) == 0


def test_override_is_transparent_for_non_listed_params():
    """A pointer NOT in ``_SCALAR_OUT_PARMS`` must be classified exactly as
    the delegated hipblas tail would — the override must not perturb any
    matrix / handle / buffer parm it doesn't explicitly target.
    """
    root = _build("""
        typedef void* hipblasLtMatmulDesc_t;
        int hipblasLtMatmul(hipblasLtMatmulDesc_t desc, const void* A, void* C);
    """)
    _bind(root, rocm.hipblaslt)
    for pname in ("A", "C"):
        p = _parm(root, "hipblasLtMatmul", pname)
        assert ("hipblasLtMatmul", pname) not in rocm.hipblaslt._SCALAR_OUT_PARMS
        assert rocm.hipblaslt.ptr_parm_intent(p) == rocm.hipblas.ptr_parm_intent(p)
        assert rocm.hipblaslt.ptr_rank(p) == rocm.hipblas.ptr_rank(p)
