# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Regression lock-in for scalar callee-allocated OUT returns in
``support.recipes.rocm.hipsparselt``.

hipSPARSELt reuses hipsparse's rules (``ptr_rank`` maps a single
pointer-to-basic-type to rank 0, and the intent chain leads with
``documented_param_intent``), which handles the workspace / compressed-size
getters whose ``@param[out]`` tags are inline and machine-parseable.

But the version/property getters (``hipsparseLtGetVersion``,
``hipsparseLtGetProperty``) are documented in ROCm's split doxygen style
(``@param[out]`` on one line, ``*  version`` on the next), which defeats
``documented_param_intent``'s tag regex — so hipSPARSELt carries a dedicated
``_SCALAR_OUT_PARMS`` override (mirroring hipBLASLt) that pins them to
``OUT_CALLEE_ALLOCATED`` + rank 0 regardless of doxygen parseability.

These tests pin both behaviors so a future change to the shared hipsparse
rank/intent rules — or a regression in the override — can't silently push
hipSPARSELt's scalar getters back into caller-allocated ``PointerTo*``
arguments. Synthetic single-function headers keep them runnable anywhere.
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
    mirroring ``interfacegen.cython._backend.initialize_nodes`` — required
    because ``documented_param_intent`` consults ``parm.ptr_rank`` to decide
    whether a documented ``[out]`` scalar is callee-allocated.
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
    intent = recipe.ptr_parm_intent(p)
    assert intent == ParmIntent.OUT_CALLEE_ALLOCATED
    assert intent.direction == ParmIntent.OUT
    assert recipe.ptr_rank(p) == 0


def test_version_and_property_are_scalar_returns():
    root = _build("""
        typedef struct { int x; } hipsparseLtHandle_t;
        typedef int hipLibraryPropertyType;
        /** @param[out] version version number. */
        int hipsparseLtGetVersion(const hipsparseLtHandle_t* handle, int* version);
        /** @param[out] value property value. */
        int hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int* value);
    """)
    _bind(root, rocm.hipsparselt)
    _assert_scalar_return(rocm.hipsparselt, _parm(root, "hipsparseLtGetVersion", "version"))
    _assert_scalar_return(rocm.hipsparselt, _parm(root, "hipsparseLtGetProperty", "value"))


def test_override_forces_scalar_return_without_parseable_tag():
    """The ``_SCALAR_OUT_PARMS`` override must force the callee-allocated
    scalar return even when the ``@param[out]`` tag is unparseable.

    ROCm documents these getters in split style (``@param[out]`` on one line,
    ``*  version`` on the next), which defeats ``documented_param_intent``'s
    tag regex. Here we drop the tags entirely to prove the override — not the
    doxygen chain — is what pins ``OUT_CALLEE_ALLOCATED`` + rank 0.
    """
    root = _build("""
        typedef struct { int x; } hipsparseLtHandle_t;
        typedef int hipLibraryPropertyType;
        int hipsparseLtGetVersion(const hipsparseLtHandle_t* handle, int* version);
        int hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int* value);
    """)
    _bind(root, rocm.hipsparselt)
    _assert_scalar_return(rocm.hipsparselt, _parm(root, "hipsparseLtGetVersion", "version"))
    _assert_scalar_return(rocm.hipsparselt, _parm(root, "hipsparseLtGetProperty", "value"))


def test_override_is_transparent_for_non_listed_params():
    """A pointer NOT in ``_SCALAR_OUT_PARMS`` must be classified exactly as
    the delegated hipsparse tail would — the override must not perturb any
    handle / descriptor parm it doesn't explicitly target.
    """
    root = _build("""
        typedef void* hipsparseLtHandle_t;
        int hipsparseLtInit(hipsparseLtHandle_t* handle);
    """)
    _bind(root, rocm.hipsparselt)
    p = _parm(root, "hipsparseLtInit", "handle")
    assert ("hipsparseLtInit", "handle") not in rocm.hipsparselt._SCALAR_OUT_PARMS
    assert rocm.hipsparselt.ptr_parm_intent(p) == rocm.hipsparse.ptr_parm_intent(p)
    assert rocm.hipsparselt.ptr_rank(p) == rocm.hipsparse.ptr_rank(p)


def test_workspace_size_is_scalar_return():
    root = _build("""
        typedef struct { int x; } hipsparseLtHandle_t;
        typedef struct { int x; } hipsparseLtMatmulPlan_t;
        /** @param[out] workspaceSize workspace size in bytes. */
        int hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t* handle,
                                          const hipsparseLtMatmulPlan_t* plan,
                                          unsigned long* workspaceSize);
    """)
    _bind(root, rocm.hipsparselt)
    _assert_scalar_return(
        rocm.hipsparselt, _parm(root, "hipsparseLtMatmulGetWorkspace", "workspaceSize")
    )


def test_compressed_sizes_are_two_scalar_returns():
    """Both size out-pointers of ``hipsparseLtSpMMACompressedSize`` become
    returned scalars (a two-element return tuple with the status)."""
    root = _build("""
        typedef struct { int x; } hipsparseLtHandle_t;
        typedef struct { int x; } hipsparseLtMatmulPlan_t;
        /**
         *  @param[out] compressedSize     size of the compressed matrix.
         *  @param[out] compressBufferSize size of the compression buffer.
         */
        int hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t* handle,
                                           const hipsparseLtMatmulPlan_t* plan,
                                           unsigned long* compressedSize,
                                           unsigned long* compressBufferSize);
    """)
    _bind(root, rocm.hipsparselt)
    for pname in ("compressedSize", "compressBufferSize"):
        _assert_scalar_return(
            rocm.hipsparselt, _parm(root, "hipsparseLtSpMMACompressedSize", pname)
        )


def test_handle_creator_is_scalar_return():
    """A ``void** handle`` creator stays a returned handle (hipsparse rule)."""
    root = _build("""
        typedef void* hipsparseLtHandle_t;
        int hipsparseLtInit(hipsparseLtHandle_t* handle);
    """)
    _bind(root, rocm.hipsparselt)
    p = _parm(root, "hipsparseLtInit", "handle")
    assert rocm.hipsparselt.ptr_parm_intent(p) == ParmIntent.OUT_CALLEE_ALLOCATED
    assert rocm.hipsparselt.ptr_rank(p) == 0
