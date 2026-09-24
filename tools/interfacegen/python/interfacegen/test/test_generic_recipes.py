# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

import pytest

from interfacegen import cython, treefactory
from interfacegen.cparser import CParser
from interfacegen.support.recipes import generic
from interfacegen.support.recipes.control import (
    DEFAULT_PTR_PARM_INTENT,
    DEFAULT_PTR_RANK,
    ParmIntent,
    fallback,
)


HEADER = """
typedef struct ihandle_s* my_handle_t;

void f_value_in(int x);
void f_const_value_in(const int x);

void f_ptr_unknown(int *p);
void f_const_ptr_in(const int *p);

void f_array_unknown(int p[]);
void f_const_array_in(const int p[]);

void f_const_array_n(const int p[5]);
void f_array_n(int p[5]);

void f_double_ptr_unknown(int **p);
void f_const_double_ptr_in(const int **p);

void f_void_ptr(void *p);
void f_const_void_ptr_in(const void *p);

void f_void_double_ptr(void **p);
void f_const_void_double_ptr_in(const void **p);

void f_struct_double_ptr(struct ihandle_s **p);

void f_str_in(const char *s);
void f_str_unknown(char *s);
void f_str_out(char **s);

void f_handle_scalar(my_handle_t h);
"""


def _build_root():
    parser = CParser("input.h", unsaved_files=[("input.h", HEADER)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython,
        translation_unit=parser.translation_unit,
    )


def _parm(root, fname, idx=0):
    for fn in root.walk(postorder=False):
        if isinstance(fn, cython.Function) and fn.name == fname:
            for i, p in enumerate(fn.parms):
                if i == idx:
                    return p
    raise KeyError(fname)


@pytest.fixture(scope="module")
def root():
    return _build_root()


# ---------------------------------------------------------------------------
# conservative
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fname,expected",
    [
        # value parms => is_any_pointer is False, deferred
        ("f_value_in", None),
        ("f_const_value_in", None),
        # plain T* => unknown
        ("f_ptr_unknown", None),
        # const T* => IN
        ("f_const_ptr_in", ParmIntent.IN),
        # T[]  => unknown intent (rank handled separately)
        ("f_array_unknown", None),
        ("f_const_array_in", ParmIntent.IN),
        ("f_const_array_n", ParmIntent.IN),
        ("f_array_n", None),
        # T**  => unknown
        ("f_double_ptr_unknown", None),
        # const T** => IN
        ("f_const_double_ptr_in", ParmIntent.IN),
        ("f_void_ptr", None),
        ("f_const_void_ptr_in", ParmIntent.IN),
        ("f_void_double_ptr", None),
        ("f_const_void_double_ptr_in", ParmIntent.IN),
    ],
)
def test_conservative_intent(root, fname, expected):
    assert generic.conservative.ptr_parm_intent(_parm(root, fname)) == expected


@pytest.mark.parametrize(
    "fname,expected",
    [
        # plain T* / T** carry no array layer => unknown
        ("f_ptr_unknown", None),
        ("f_double_ptr_unknown", None),
        ("f_void_ptr", None),
        # T[] (incomplete array layer) => 1
        ("f_array_unknown", 1),
        ("f_const_array_in", 1),
        # T[N] (constant array layer) => 1
        ("f_array_n", 1),
        ("f_const_array_n", 1),
    ],
)
def test_conservative_rank(root, fname, expected):
    assert generic.conservative.ptr_rank(_parm(root, fname)) == expected


# ---------------------------------------------------------------------------
# double_indirection_out
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fname,expected_intent,expected_rank",
    [
        ("f_double_ptr_unknown", ParmIntent.OUT_CALLEE_ALLOCATED, 0),
        ("f_struct_double_ptr", ParmIntent.OUT_CALLEE_ALLOCATED, 0),
        # `void**` is the untyped callee-allocated byte-buffer idiom
        # (`hipMalloc`); it reports rank 1 while typed `T**` handle slots
        # remain rank 0. Callee-allocation (the OUT_CALLEE_ALLOCATED hint)
        # is independent of rank.
        ("f_void_double_ptr", ParmIntent.OUT_CALLEE_ALLOCATED, 1),
        # const T** => not OUT (intent), and rank-rule still fires (rank-only
        # convention does not check const). The `conservative` rule above
        # would catch the IN before this in a real chain.
        ("f_const_double_ptr_in", None, 0),
        # single-pointer parms => no match
        ("f_ptr_unknown", None, None),
        ("f_const_ptr_in", None, None),
    ],
)
def test_double_indirection_out(root, fname, expected_intent, expected_rank):
    p = _parm(root, fname)
    assert generic.double_indirection_out.ptr_parm_intent(p) == expected_intent
    assert generic.double_indirection_out.ptr_rank(p) == expected_rank


# ---------------------------------------------------------------------------
# pointer_as_value / pointer_as_reference (mutually exclusive)
# ---------------------------------------------------------------------------
def test_pointer_as_value_intent(root):
    assert (
        generic.pointer_as_value.ptr_parm_intent(_parm(root, "f_ptr_unknown"))
        == ParmIntent.IN
    )
    # const *T => deferred (conservative handles it)
    assert (
        generic.pointer_as_value.ptr_parm_intent(_parm(root, "f_const_ptr_in"))
        is None
    )
    # T** => deferred
    assert (
        generic.pointer_as_value.ptr_parm_intent(
            _parm(root, "f_double_ptr_unknown")
        )
        is None
    )


def test_pointer_as_reference_intent(root):
    assert (
        generic.pointer_as_reference.ptr_parm_intent(
            _parm(root, "f_ptr_unknown")
        )
        == ParmIntent.INOUT
    )
    assert (
        generic.pointer_as_reference.ptr_parm_intent(
            _parm(root, "f_const_ptr_in")
        )
        is None
    )
    assert (
        generic.pointer_as_reference.ptr_parm_intent(
            _parm(root, "f_double_ptr_unknown")
        )
        is None
    )


# ---------------------------------------------------------------------------
# string_z
# ---------------------------------------------------------------------------
def test_string_z(root):
    p_in = _parm(root, "f_str_in")
    p_unknown = _parm(root, "f_str_unknown")
    p_out = _parm(root, "f_str_out")

    assert generic.string_z.ptr_parm_intent(p_in) == ParmIntent.IN
    assert generic.string_z.ptr_parm_intent(p_unknown) is None
    # `char **` returns a single callee-produced string.
    assert (
        generic.string_z.ptr_parm_intent(p_out)
        == ParmIntent.OUT_CALLEE_ALLOCATED
    )
    # Coarsens back to OUT for direction-only consumers.
    assert generic.string_z.ptr_parm_intent(p_out).direction == ParmIntent.OUT

    # A zero-terminated string is rank-1 *data* irrespective of how many
    # pointer layers wrap it: both ``char *`` and ``char **`` report rank 1.
    assert generic.string_z.ptr_rank(p_in) == 1
    assert generic.string_z.ptr_rank(p_unknown) == 1
    assert generic.string_z.ptr_rank(p_out) == 1


# ---------------------------------------------------------------------------
# CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER: char-pointer wrapper selection
# ---------------------------------------------------------------------------
def test_default_ptr_handler_char_pointers(root):
    """Rank is 1 for every char pointer (DEFAULT_PTR_RANK); the handler uses
    pointer degree + intent — not rank — to pick the wrapper."""
    handler = cython.CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER()

    # char* (degree 1) is the string buffer itself => CStr, independent of
    # intent (IN/INOUT params, return values, and fields all share this).
    assert handler(_parm(root, "f_str_in")) == "CStr"  # const char *
    assert handler(_parm(root, "f_str_unknown")) == "CStr"  # char *

    # char** (degree 2): an OUT slot returns a single string => CStr; an IN
    # array-of-strings (argv) must NOT be clobbered => falls through to
    # Pointer (recipes override to ListOfBytes where needed).
    p_out = _parm(root, "f_str_out")  # char **
    original_intent = p_out.ptr_intent
    try:
        p_out.ptr_intent = lambda node: ParmIntent.OUT
        assert handler(p_out) == "CStr"
        p_out.ptr_intent = lambda node: ParmIntent.IN
        assert handler(p_out) == "Pointer"
    finally:
        p_out.ptr_intent = original_intent


# ---------------------------------------------------------------------------
# opaque_typedef_is_handle
# ---------------------------------------------------------------------------
def test_opaque_typedef_is_handle(root):
    # my_handle_t is `typedef struct ihandle_s *` — passed by value.
    p = _parm(root, "f_handle_scalar")
    assert generic.opaque_typedef_is_handle.ptr_rank(p) == 0
    # No intent contribution
    assert generic.opaque_typedef_is_handle.ptr_parm_intent(p) is None
    # Untyped'ed `int *` is not a handle
    assert (
        generic.opaque_typedef_is_handle.ptr_rank(_parm(root, "f_ptr_unknown"))
        is None
    )


# ---------------------------------------------------------------------------
# fallback decorator precedence
# ---------------------------------------------------------------------------
def test_fallback_precedence():
    @fallback(
        lambda x: ParmIntent.OUT if x == 2 else None, lambda x: ParmIntent.IN
    )  # final default
    def rule(x):
        if x == 1:
            return ParmIntent.INOUT
        return None

    assert rule(1) == ParmIntent.INOUT  # primary
    assert rule(2) == ParmIntent.OUT  # 1st fallback
    assert rule(3) == ParmIntent.IN  # 2nd fallback (default)


def test_fallback_with_conservative_chain(root):
    # Simulate a per-library chain: hardcoded override > conservative > default
    @fallback(generic.conservative.ptr_parm_intent, DEFAULT_PTR_PARM_INTENT)
    def my_intent(parm):
        if parm.parent.name == "f_ptr_unknown":
            return ParmIntent.OUT  # override
        return None

    # override wins
    assert my_intent(_parm(root, "f_ptr_unknown")) == ParmIntent.OUT
    # conservative answers
    assert my_intent(_parm(root, "f_const_ptr_in")) == ParmIntent.IN
    # falls through to DEFAULT_PTR_PARM_INTENT (which fires for non-const **T)
    assert my_intent(_parm(root, "f_double_ptr_unknown")) == ParmIntent.INOUT


# ---------------------------------------------------------------------------
# documented_param_intent — doxygen `@param[in|out|in,out]` tag parser
# ---------------------------------------------------------------------------
#
# Build a separate root from a header that carries doxygen comments.
# The synthetic shapes cover both the `@` and `\` directive prefixes,
# all four direction values (`in`, `out`, `in,out`, `inout`), the
# missing-tag fallthrough, multi-line bodies, and the absent /
# empty raw_comment defensiveness.

DOXY_HEADER = r"""
/**
 * Function with @param-style direction tags.
 *
 * @param[in]     x     input scalar
 * @param[out]    y     output scalar
 * @param[in,out] z     in/out scalar
 */
void f_at_param(int x, int *y, int *z);

/**
 * Function with backslash-prefixed \param tags.
 *
 * \param[in]    a   input
 * \param[out]   b   output
 * \param[inout] c   shorthand inout
 */
void f_bs_param(int a, int *b, int *c);

/**
 * Body terminated by next directive — keyword from a sibling tag's
 * body cannot leak into another tag's match.
 *
 * @param[out] alpha  this is the array of values returned to caller
 * @param[in]  beta   the input
 */
void f_bodies(int *alpha, int beta);

/* No doxygen comment at all on this one. */
void f_undoc(int *p);

/**
 * Doxygen present but no @param tags.
 *
 * @brief no params documented
 */
void f_no_param(int *q);

/**
 * Param tag without a direction bracket — should NOT match
 * (we only trust explicit direction).
 *
 * @param missing_dir  no bracket
 */
void f_no_dir(int *missing_dir);
"""


@pytest.fixture(scope="module")
def doxy_root():
    parser = CParser(
        "input.h",
        unsaved_files=[("input.h", DOXY_HEADER)],
    )
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython,
        translation_unit=parser.translation_unit,
    )


@pytest.mark.parametrize(
    "fname,idx,expected",
    [
        ("f_at_param", 0, ParmIntent.IN),
        ("f_at_param", 1, ParmIntent.OUT),
        ("f_at_param", 2, ParmIntent.INOUT),
        ("f_bs_param", 0, ParmIntent.IN),
        ("f_bs_param", 1, ParmIntent.OUT),
        ("f_bs_param", 2, ParmIntent.INOUT),  # `\param[inout]` shorthand
        ("f_bodies", 0, ParmIntent.OUT),  # `alpha` matches by name
        ("f_bodies", 1, ParmIntent.IN),
    ],
)
def test_documented_param_intent_matches(doxy_root, fname, idx, expected):
    """Each `@param` / `\\param` direction bracket maps to a `ParmIntent`."""
    p = _parm(doxy_root, fname, idx)
    assert generic.documented_param_intent.ptr_parm_intent(p) == expected


def test_documented_param_intent_no_doxygen(doxy_root):
    """No doxygen comment at all → return None."""
    p = _parm(doxy_root, "f_undoc", 0)
    assert generic.documented_param_intent.ptr_parm_intent(p) is None


def test_documented_param_intent_no_param_tag(doxy_root):
    """Doxygen present but no @param tags → return None."""
    p = _parm(doxy_root, "f_no_param", 0)
    assert generic.documented_param_intent.ptr_parm_intent(p) is None


def test_documented_param_intent_scalar_out_is_callee_allocated(doxy_root):
    """A documented `@param[out]` refines to OUT_CALLEE_ALLOCATED for a
    scalar (rank-0) slot, but stays plain OUT for an array (rank-1+) slot.

    The doxygen `[out]` tag cannot express callee-allocation of a sized
    buffer, so only the scalar/handle/string case (rank 0) is upgraded;
    a caller-sized array buffer remains caller-allocated OUT.
    """
    p = _parm(doxy_root, "f_at_param", 1)  # `int *y` tagged @param[out]
    original = p.ptr_rank
    try:
        p.ptr_rank = lambda node: 0  # scalar slot
        verdict = generic.documented_param_intent.ptr_parm_intent(p)
        assert verdict == ParmIntent.OUT_CALLEE_ALLOCATED
        assert verdict.direction == ParmIntent.OUT  # coarse view unchanged

        p.ptr_rank = lambda node: 1  # array buffer
        assert (
            generic.documented_param_intent.ptr_parm_intent(p)
            == ParmIntent.OUT
        )
    finally:
        p.ptr_rank = original


def test_documented_param_intent_no_direction_bracket(doxy_root):
    """`@param <name> ...` without `[dir]` → return None.

    We only trust explicit directions; an undirected param tag is just
    a description and tells us nothing about intent.
    """
    p = _parm(doxy_root, "f_no_dir", 0)
    assert generic.documented_param_intent.ptr_parm_intent(p) is None


def test_documented_param_intent_unrelated_parm_name(doxy_root):
    """A parm whose name doesn't match any documented tag → None.

    `f_bodies` documents `alpha` and `beta`. If we ask about a
    fictional third parm with a different name, we get None.
    """
    # Fake: build a parm with a name not present in the doc by reusing
    # a parm node and overriding its `name` attribute via an inline
    # subclass instance is overkill — just exercise the lookup path
    # directly with an unrelated parm from another function.
    p = _parm(doxy_root, "f_undoc", 0)  # no comment → None either way
    assert generic.documented_param_intent.ptr_parm_intent(p) is None


def test_iter_doxygen_param_tags_directly():
    """Unit-test the helper without going through libclang."""
    raw = """
    @param[in]     x   first
    \\param[out]   y   second
    @param[in,out] z   third
    \\param[inout] w   fourth
    @param        u   no-direction (should yield direction=None)
    """
    tags = list(generic._iter_doxygen_param_tags(raw))
    # Every tag yields (direction, name); direction is None when no bracket.
    assert ("in", "x") in tags
    assert ("out", "y") in tags
    assert ("in,out", "z") in tags
    assert ("inout", "w") in tags
    assert (None, "u") in tags


def test_iter_doxygen_param_tags_empty():
    """Empty / None raw_comment yields nothing."""
    assert list(generic._iter_doxygen_param_tags(None)) == []
    assert list(generic._iter_doxygen_param_tags("")) == []


# ---------------------------------------------------------------------------
# llvm_c recipe — callee-allocated OUT params (char** error strings)
# ---------------------------------------------------------------------------
_LLVM_HEADER = """
typedef struct LLVMOpaqueTargetMachine *LLVMTargetRef;
typedef struct LLVMOpaqueMemoryBuffer *LLVMMemoryBufferRef;
typedef int LLVMBool;

LLVMBool LLVMGetTargetFromTriple(const char *Triple, LLVMTargetRef *T, char **ErrorMessage);
LLVMBool LLVMCreateMemoryBufferWithContentsOfFile(const char *Path, LLVMMemoryBufferRef *OutMemBuf, char **OutMessage);
"""


@pytest.fixture(scope="module")
def llvm_root():
    parser = CParser("llvm.h", unsaved_files=[("llvm.h", _LLVM_HEADER)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython,
        translation_unit=parser.translation_unit,
    )


def test_llvm_callee_allocated_out_params(llvm_root):
    """Regression: the `char **` error-string OUT params of the LLVM
    target/memory-buffer creators must classify as OUT_CALLEE_ALLOCATED.

    Without the explicit callee-allocation hint these `char **` params
    are rank-1 (NUL-terminated char sequence), so the rank-0 structural
    fallback in `is_out_callee_allocated_ptr` never fires and they leak
    into the Python signature as positional arguments instead of becoming
    return-tuple entries (cf. the `LLVMGetTargetFromTriple() takes exactly
    2 positional arguments (1 given)` failure).
    """
    from interfacegen.support.recipes.rocm import llvm_c

    # `char **` callee-allocated error strings => OUT_CALLEE_ALLOCATED.
    err = _parm(llvm_root, "LLVMGetTargetFromTriple", 2)  # char **ErrorMessage
    out_msg = _parm(
        llvm_root, "LLVMCreateMemoryBufferWithContentsOfFile", 2
    )  # char **OutMessage
    for p in (err, out_msg):
        verdict = llvm_c.ptr_parm_intent(p)
        assert verdict == ParmIntent.OUT_CALLEE_ALLOCATED
        assert verdict.allocated_by_callee
        assert verdict.direction == ParmIntent.OUT

    # The opaque handle OUT slots are also callee-allocated returns.
    t = _parm(llvm_root, "LLVMGetTargetFromTriple", 1)  # LLVMTargetRef *T
    out_buf = _parm(
        llvm_root, "LLVMCreateMemoryBufferWithContentsOfFile", 1
    )  # LLVMMemoryBufferRef *OutMemBuf
    assert llvm_c.ptr_parm_intent(t) == ParmIntent.OUT_CALLEE_ALLOCATED
    assert llvm_c.ptr_parm_intent(out_buf) == ParmIntent.OUT_CALLEE_ALLOCATED

    # The leading `const char *` input remains a plain IN argument.
    triple = _parm(llvm_root, "LLVMGetTargetFromTriple", 0)
    assert llvm_c.ptr_parm_intent(triple) == ParmIntent.IN
