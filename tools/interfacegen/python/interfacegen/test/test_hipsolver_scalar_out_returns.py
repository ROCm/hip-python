# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression lock-in for hipSOLVER's ``*_bufferSize`` workspace-size
callee-allocated OUT returns in ``support.recipes.rocm.hipsolver``.

hipSOLVER's ``*_bufferSize`` queries write the required workspace *byte
count* to a host ``int*`` / ``size_t*`` named ``lwork``. The 64-bit
``hipsolverDnX*`` variants split it into two host size outputs,
``lworkOnDevice`` + ``lworkOnHost`` (both host ``size_t*`` receiving a byte
count — NOT device addresses; the genuine device buffer is the by-value /
``void*`` ``workOnDevice`` of the compute call).

The recipe carries a ``_is_buffersize_lwork`` override that pins any pointer
named ``lwork*`` inside a ``*_bufferSize`` function to
``OUT_CALLEE_ALLOCATED`` + rank 0 so it reads as a Python return value.
These tests pin that behavior — and pin that it stays scoped: the by-value
``int lwork`` *input* of the compute calls, the device buffers (``work``,
``devIpiv``), and ``lwork``-named pointers outside ``*_bufferSize`` must be
untouched. Synthetic single-function headers keep them runnable anywhere.
"""

from interfacegen import cython, treefactory
from interfacegen.cparser import CParser
from interfacegen.support.recipes import rocm
from interfacegen.support.recipes.control import ParmIntent


def _build(header_text: str):
    parser = CParser("input.h", unsaved_files=[("input.h", header_text)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython,
        translation_unit=parser.translation_unit,
    )


def _bind(root, recipe):
    """Attach the recipe's ``ptr_rank`` / intent callbacks to every Parm,
    mirroring ``interfacegen.cython._backend.initialize_nodes``.
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


def test_int_lwork_buffersize_is_scalar_return():
    """``int* lwork`` in a ``*_bufferSize`` query -> returned scalar."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        int hipsolverDgetrf_bufferSize(hipsolverHandle_t* handle,
                                       int m, int n, double* A, int lda,
                                       int* lwork);
    """
    )
    _bind(root, rocm.hipsolver)
    _assert_scalar_return(
        rocm.hipsolver, _parm(root, "hipsolverDgetrf_bufferSize", "lwork")
    )


def test_size_t_lwork_buffersize_is_scalar_return():
    """``size_t* lwork`` variant likewise becomes a returned scalar."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        typedef unsigned long size_t;
        int hipsolverDnDgetrf_bufferSize(hipsolverHandle_t* handle,
                                         int m, int n, double* A, int lda,
                                         size_t* lwork);
    """
    )
    _bind(root, rocm.hipsolver)
    _assert_scalar_return(
        rocm.hipsolver, _parm(root, "hipsolverDnDgetrf_bufferSize", "lwork")
    )


def test_dnx_dual_lwork_on_device_and_host_both_convert():
    """The 64-bit ``hipsolverDnX*_bufferSize`` dual host size outputs
    ``lworkOnDevice`` + ``lworkOnHost`` both become returned scalars
    (a three-element return tuple with the status)."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        typedef unsigned long size_t;
        int hipsolverDnXgetrf_bufferSize(hipsolverHandle_t* handle,
                                         int m, int n, double* A, int lda,
                                         size_t* lworkOnDevice,
                                         size_t* lworkOnHost);
    """
    )
    _bind(root, rocm.hipsolver)
    for pname in ("lworkOnDevice", "lworkOnHost"):
        _assert_scalar_return(
            rocm.hipsolver, _parm(root, "hipsolverDnXgetrf_bufferSize", pname)
        )


def test_lwork_outside_buffersize_is_not_converted():
    """An ``int* lwork`` in a function NOT ending in ``_bufferSize`` must
    stay a caller-allocated buffer (the scope guard rejects it)."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        int hipsolverDgetrf_query(hipsolverHandle_t* handle,
                                  int m, int n, double* A, int lda,
                                  int* lwork);
    """
    )
    _bind(root, rocm.hipsolver)
    p = _parm(root, "hipsolverDgetrf_query", "lwork")
    assert not rocm.hipsolver._is_buffersize_lwork(p)
    assert rocm.hipsolver.ptr_parm_intent(p) != ParmIntent.OUT_CALLEE_ALLOCATED
    assert rocm.hipsolver.ptr_rank(p) == hipblas_rank(p)


def test_by_value_lwork_input_untouched():
    """The by-value ``int lwork`` *input* of a compute call is not a
    pointer and must not be classified as a scalar OUT return."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        int hipsolverDgetrf(hipsolverHandle_t* handle,
                            int m, int n, double* A, int lda,
                            double* work, int lwork,
                            int* devIpiv, int* devInfo);
    """
    )
    _bind(root, rocm.hipsolver)
    p = _parm(root, "hipsolverDgetrf", "lwork")
    assert not rocm.hipsolver._is_buffersize_lwork(p)
    assert rocm.hipsolver.ptr_parm_intent(p) != ParmIntent.OUT_CALLEE_ALLOCATED


def test_other_buffersize_pointer_untouched():
    """A non-``lwork`` pointer in a ``*_bufferSize`` (e.g. ``devIpiv``) is
    left to the delegated hipblas rank/intent — the override is scoped to
    ``lwork*`` names only."""
    root = _build(
        """
        typedef struct { int x; } hipsolverHandle_t;
        int hipsolverDgetrf_bufferSize(hipsolverHandle_t* handle,
                                       int m, int n, double* A, int lda,
                                       int* devIpiv, int* lwork);
    """
    )
    _bind(root, rocm.hipsolver)
    devipiv = _parm(root, "hipsolverDgetrf_bufferSize", "devIpiv")
    assert not rocm.hipsolver._is_buffersize_lwork(devipiv)
    assert rocm.hipsolver.ptr_rank(devipiv) == hipblas_rank(devipiv)
    # ...while the sibling lwork in the same function still converts.
    _assert_scalar_return(
        rocm.hipsolver, _parm(root, "hipsolverDgetrf_bufferSize", "lwork")
    )


def hipblas_rank(p):
    """The rank hipSOLVER would delegate to for a non-``lwork`` pointer."""
    return rocm.hipblas.ptr_rank(p)
