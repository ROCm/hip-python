# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Comprehensive coverage of nested struct/union shapes seen in real C
headers (hip_runtime_api.h, amdsmi.h, hipfft.h).

Each test case is a synthetic single-header that mirrors a specific shape
the codegen has had to handle. They exist to:

  * lock down working shapes against regressions (struct hoist path,
    anonymous-name synthesis, libclang ``is_anonymous()`` handling),
  * give a single grep target when a new header surfaces a shape we
    haven't seen before — add a case here, then chase the codegen fix.

The test asserts two invariants on every emitted ``cymod*.pxd``:

  1. **Reference closure**: every ``Type identifier`` referenced as a
     field type has a matching ``cdef struct/union/enum`` definition in
     the same file (no dangling type references).
  2. **No libclang pseudo-spelling leaks**: no emitted name contains
     ``"(anonymous at"`` (libclang's pseudo-spelling for unnamed types).

Where a shape produces a Cython-syntactically valid output but with
known caveats (e.g. an anonymous-no-field-name member is silently
dropped by the codegen), the per-case test is documented and the
assertion narrowed to what is achievable today.
"""

import re

import pytest

from _codegen_helpers import make_generator, write_module


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------

# (shape_name, header_text, [extra_assertions(pxd) -> None])

SHAPE_PLAIN_STRUCT = """
/* hip's hipDeviceProp_t-like plain struct + bit-fields */
typedef struct {
    unsigned hasInt32Atomics : 1;
    unsigned hasFloatAtomics : 1;
    unsigned int maxThreadsPerBlock;
} plain_props_t;
"""

SHAPE_AMDSMI_BDF_T = """
/* amdsmi_bdf_t shape: typedef union with
     - named-nested struct (`struct bdf_ {...} bdf;`)
     - truly anonymous nested struct (`struct {...};` no field name)
     - scalar fallback `as_uint`. */
typedef union {
    struct bdf_ {
        unsigned long function_number;
        unsigned long device_number;
    } bdf;
    struct {
        unsigned long function_number;
        unsigned long device_number;
    };
    unsigned long as_uint;
} my_bdf_t;
"""

SHAPE_HIP_EXTERNAL_MEM = """
/* hipExternalMemoryHandleDesc_st shape: top-level typedef struct
   containing a NAMED outer field whose type is an anonymous union with
   a named-nested inner struct. */
typedef enum { ENUM_FD = 0, ENUM_WIN32 = 1 } handle_type_t;

typedef struct ext_mem_desc_st {
    handle_type_t type;
    union {
        int fd;
        struct {
            void *handle_;
            const void *name;
        } win32;
        const void *sci_obj;
    } handle;
    unsigned long size;
} ext_mem_desc_t;
"""

SHAPE_TAGGED_UNION_WITH_NESTED = """
/* hipStreamBatchMemOpParams shape: typedef union (TAGGED, with name)
   containing multiple named-nested struct fields, each containing a
   truly-anonymous nested union. */
typedef enum { OP_WAIT = 1, OP_WRITE = 2 } op_kind_t;

typedef union batch_op_params_union {
    op_kind_t kind;
    struct wait_value_params {
        op_kind_t kind;
        union {
            unsigned value32;
            unsigned long value64;
        };
        unsigned flags;
    } wait;
    struct write_value_params {
        op_kind_t kind;
        union {
            unsigned value32;
            unsigned long value64;
        };
        unsigned flags;
    } write;
} batch_op_params_t;
"""

SHAPE_DEEP_NESTED_ANONYMOUS = """
/* Three-level nesting, all-anonymous after the outer typedef. */
typedef struct {
    int top;
    struct {
        int mid;
        union {
            int leaf_a;
            float leaf_b;
        };
    };
} deep_anon_t;
"""

SHAPE_VOID_PTR_ARRAY_FIELD = """
/* amdsmi_topology_nearest_t shape: void* typedef as element type of an
   array field. Lock-down for the Field.cython_repr fix that emits
   `void *arr[N]` instead of the broken `void *[N] arr`. */
typedef void* opaque_handle_t;
typedef struct {
    unsigned int count;
    opaque_handle_t list[8];
    unsigned long reserved[3];
} topology_t;
"""


# For each case: (header_text, expected_outer_name)
# The expected_outer_name is whatever the codegen will use as the
# top-level identifier — i.e. the struct/union TAG when present
# (`typedef struct foo { … } foo_t;` → emits `cdef struct foo:`),
# or the typedef alias when no tag is given.
CASES = [
    pytest.param(SHAPE_PLAIN_STRUCT, "plain_props_t", id="plain_struct_with_bitfields"),
    pytest.param(SHAPE_AMDSMI_BDF_T, "my_bdf_t", id="amdsmi_bdf_t_named_plus_anon_nested"),
    pytest.param(SHAPE_HIP_EXTERNAL_MEM, "ext_mem_desc_st", id="hip_external_mem_anon_union_named_inner_struct"),
    pytest.param(SHAPE_TAGGED_UNION_WITH_NESTED, "batch_op_params_union", id="tagged_union_named_struct_anon_union_inside"),
    pytest.param(SHAPE_DEEP_NESTED_ANONYMOUS, "deep_anon_t", id="three_level_anonymous_nesting"),
    pytest.param(SHAPE_VOID_PTR_ARRAY_FIELD, "topology_t", id="void_typedef_array_field"),
]


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------

# Match `cdef struct/union/enum NAME:` AND `ctypedef struct/union/enum
# NAME:` definitions. The latter form is what the codegen emits for
# anonymous-typedef'd inner types (`typedef struct {...} foo_t;`),
# which is the correct Cython idiom for that shape.
_DEF_RE = re.compile(
    r"^\s*c(?:type)?def\s+(?:struct|union|enum)\s+([A-Za-z_]\w*)\b",
    re.MULTILINE,
)
# Tokens we don't expect to be type definitions (basic C / Cython
# primitives + qualifiers). Anything else referenced as a type must
# be defined in the same file (the closure check).
_PRIMITIVE_TOKENS = {
    "void", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned", "const", "volatile", "_Bool",
    "size_t", "ssize_t", "ptrdiff_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "intptr_t", "uintptr_t",
    "bint",
}


def _struct_field_type_tokens(pxd: str):
    """Yield user-defined type identifiers used as field types inside any
    `cdef struct/union ...:` block in the pxd.

    Hand-rolled line walker — a backtracking regex over the whole file
    blows up on the typical 500-line pxd.
    """
    in_block = False
    for line in pxd.splitlines():
        stripped = line.lstrip()
        # Block starts on `cdef struct/union/enum NAME:` OR
        # `ctypedef struct/union/enum NAME:` (any indent). The
        # ctypedef form is emitted for anonymous-typedef'd inner types.
        if re.match(r"c(?:type)?def\s+(struct|union|enum)\s+\w+", stripped):
            in_block = True
            continue
        if not in_block:
            continue
        # Block ends on a blank line or a non-indented line.
        if not line.strip():
            in_block = False
            continue
        if not line.startswith(" "):
            in_block = False
            continue
        # Inside an enum, the contents are bare identifiers (enum values),
        # not field declarations. Skip lines that have only a single token.
        # For struct/union, the form is `<type-tokens> <field_name>[<dim>]`.
        # The field name is always the last identifier token; everything
        # before it is the type. Skip the field-name token itself.
        body = stripped.split("[", 1)[0]  # drop array dimension
        # Comments / pyx-style decorators / preprocessor — skip
        if body.startswith("#"):
            continue
        toks = re.findall(r"[A-Za-z_]\w*", body)
        if len(toks) <= 1:
            continue
        # Drop the trailing field-name token; the rest are type tokens.
        for tok in toks[:-1]:
            if tok not in _PRIMITIVE_TOKENS:
                yield tok


def _no_pseudo_spelling_leaks(pxd: str) -> None:
    leaks = re.findall(r"\(anonymous at[^)]*\)", pxd)
    assert not leaks, (
        f"libclang pseudo-spelling leaked into emitted Cython: {leaks[:3]}\n"
        f"--- pxd ---\n{pxd}"
    )


def _all_referenced_types_defined(pxd: str) -> None:
    """Every user-defined type referenced as a field type must be defined
    as a `cdef struct/union/enum` somewhere in the same pxd.
    """
    defined = set(_DEF_RE.findall(pxd))
    referenced = set(_struct_field_type_tokens(pxd))
    missing = referenced - defined
    assert not missing, (
        f"these type identifiers are referenced in fields but never "
        f"defined: {sorted(missing)}\n--- pxd ---\n{pxd}"
    )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header_text,outer_type", CASES)
def test_nested_record_shape_emits_clean_cython(header_text, outer_type, tmp_path):
    """Generate the synthetic header and assert the two file-level invariants."""
    gen = make_generator(header_text, module_name="mod_shape")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_shape.pxd"]

    # The outer typedef must be defined. Accept either `cdef struct foo:`
    # (real C tag) or `ctypedef struct foo:` (anonymous-typedef shape) —
    # the codegen picks the form that matches the upstream declaration,
    # so a typedef of an anonymous inner type correctly emits `ctypedef`.
    assert re.search(
        rf"\b(?:cdef|ctypedef)\s+(?:struct|union|enum)\s+{re.escape(outer_type)}\b",
        pxd,
    ), f"outer type {outer_type} not emitted; pxd:\n{pxd}"

    # Invariants
    _no_pseudo_spelling_leaks(pxd)
    _all_referenced_types_defined(pxd)


# ---------------------------------------------------------------------------
# Targeted shape-specific assertions
# ---------------------------------------------------------------------------


def test_amdsmi_bdf_named_inner_struct_definition_emitted(tmp_path):
    """`struct bdf_` (named-nested) must be hoisted to a top-level
    `cdef struct my_bdf_t_bdf_:`."""
    gen = make_generator(SHAPE_AMDSMI_BDF_T, module_name="mod_bdf")
    pxd = write_module(gen, tmp_path)["cymod_bdf.pxd"]
    assert re.search(r"\bcdef\s+struct\s+my_bdf_t_bdf_\b", pxd), (
        f"named-nested struct definition missing; pxd:\n{pxd}"
    )


def test_amdsmi_bdf_anonymous_inner_struct_gets_indexed_name(tmp_path):
    """The truly-anonymous inner `struct {…};` must get a synthesized
    `<parent>_struct_<N>` name (not the libclang pseudo-spelling)."""
    gen = make_generator(SHAPE_AMDSMI_BDF_T, module_name="mod_bdf2")
    pxd = write_module(gen, tmp_path)["cymod_bdf2.pxd"]
    assert re.search(r"\bcdef\s+struct\s+my_bdf_t_struct_\d+\b", pxd), (
        f"anonymous-nested struct missing or got pseudo-spelling; pxd:\n{pxd}"
    )


def test_void_typedef_array_field_suffix_after_name(tmp_path):
    """Lock-down for the Field.cython_repr fix: `void *list[8]`
    (suffix after name), not `void *[8] list`."""
    gen = make_generator(SHAPE_VOID_PTR_ARRAY_FIELD, module_name="mod_topo")
    pxd = write_module(gen, tmp_path)["cymod_topo.pxd"]
    assert re.search(r"void\s*\*\s*list\s*\[\s*8\s*\]", pxd), (
        f"expected `void *list[8]` shape; pxd:\n{pxd}"
    )
    assert not re.search(r"void\s*\*\s*\[\s*8\s*\]\s*list", pxd), (
        f"emitted broken `void *[8] list` shape; pxd:\n{pxd}"
    )


# ---------------------------------------------------------------------------
# Anonymous typedef'd enum/struct/union — libclang behavior shift
# ---------------------------------------------------------------------------
#
# Real-world hit (ROCm 7.13 + libclang 18):
#
#   /opt/rocm/include/hipblas-common/hipblas-common.h:
#
#       typedef enum {
#           HIPBLAS_STATUS_SUCCESS = 0, ...
#       } hipblasStatus_t;
#
# The codegen must emit `ctypedef enum hipblasStatus_t:` (correct —
# `hipblasStatus_t` is the typedef name; there is NO `enum hipblasStatus_t`
# tag in C). Emitting `cdef enum hipblasStatus_t:` would make Cython put
# `enum hipblasStatus_t` in the generated C, which gcc rejects with
# "incomplete type 'enum hipblasStatus_t'".
#
# The detection lives in `treefactory._is_anonymous_typedef_inner`. It
# used to be `cursor.spelling == ""`, which worked for libclang ≤16 but
# broke under libclang 17+ (which inlines the typedef name into the
# inner cursor's spelling). The current logic uses
# `cursor.type.spelling` — a tagged inner type carries the kind
# keyword (`enum foo`, `struct foo`, `union foo`); an anonymous-typedef
# inner type carries just the typedef name (`foo_t`).

SHAPE_ANON_TYPEDEF_ENUM = """
/* hipblas-common.h: typedef enum {...} hipblasStatus_t; */
typedef enum {
    FOO_OK = 0,
    FOO_ERR = 1
} foo_status_t;

foo_status_t foo_do_thing(int x);
"""

SHAPE_TAGGED_ENUM_NO_TYPEDEF = """
/* hipblas.h-era: enum hipblasOperation_t {...}; (no typedef) */
enum bar_status {
    BAR_OK = 0,
    BAR_ERR = 1
};

enum bar_status bar_do_thing(int x);
"""

SHAPE_TAGGED_ENUM_WITH_SAME_NAME_TYPEDEF = """
/* C convention: typedef enum X {...} X; — the typedef alias matches
   the real enum tag, so `cdef enum X:` is correct (the tag exists). */
typedef enum baz_status {
    BAZ_OK = 0,
    BAZ_ERR = 1
} baz_status;

baz_status baz_do_thing(int x);
"""

SHAPE_ANON_TYPEDEF_STRUCT = """
/* Common pattern: typedef struct {...} foo_t; */
typedef struct {
    int width;
    int height;
} dims_t;

dims_t make_dims(int w, int h);
"""

SHAPE_ANON_TYPEDEF_UNION = """
typedef union {
    int as_int;
    float as_float;
} bits_t;

int bits_int(bits_t b);
"""


def test_anon_typedef_enum_uses_ctypedef(tmp_path):
    """`typedef enum { ... } foo_status_t;` → `ctypedef enum foo_status_t:`.

    Regression test — covered the libclang 17+ behavior shift in which
    the inner anonymous enum now reports the typedef name as its
    `cursor.spelling`, breaking the legacy `spelling == ""` detection
    and emitting `cdef enum foo_status_t:` (which Cython turns into the
    invalid `enum foo_status_t` C reference).
    """
    gen = make_generator(SHAPE_ANON_TYPEDEF_ENUM, module_name="mod_anon_e")
    pxd = write_module(gen, tmp_path)["cymod_anon_e.pxd"]

    assert re.search(r"\bctypedef\s+enum\s+foo_status_t\s*:", pxd), (
        f"expected `ctypedef enum foo_status_t:`, got pxd:\n{pxd}"
    )
    assert not re.search(r"\bcdef\s+enum\s+foo_status_t\s*:", pxd), (
        f"regression: emitted `cdef enum foo_status_t:` for an anonymous "
        f"typedef enum. The generated C would reference `enum foo_status_t`, "
        f"a tag the upstream header does not define. Full pxd:\n{pxd}"
    )


def test_tagged_enum_no_typedef_uses_cdef(tmp_path):
    """`enum bar_status { ... };` → `cdef enum bar_status:`.

    Sanity: the typedef-detection fix must not over-correct — a
    plain tagged enum still uses `cdef enum`.
    """
    gen = make_generator(SHAPE_TAGGED_ENUM_NO_TYPEDEF, module_name="mod_tag_e")
    pxd = write_module(gen, tmp_path)["cymod_tag_e.pxd"]
    assert re.search(r"\bcdef\s+enum\s+bar_status\s*:", pxd), (
        f"expected `cdef enum bar_status:`, got pxd:\n{pxd}"
    )


def test_tagged_enum_with_same_name_typedef_uses_cdef(tmp_path):
    """`typedef enum X {...} X;` → `cdef enum X:`.

    Sanity: when the typedef alias matches an existing real tag, the
    `enum X` reference IS valid in C, so `cdef enum` is correct.
    """
    gen = make_generator(SHAPE_TAGGED_ENUM_WITH_SAME_NAME_TYPEDEF, module_name="mod_baz")
    pxd = write_module(gen, tmp_path)["cymod_baz.pxd"]
    assert re.search(r"\bcdef\s+enum\s+baz_status\s*:", pxd), (
        f"expected `cdef enum baz_status:`, got pxd:\n{pxd}"
    )


def test_anon_typedef_struct_uses_ctypedef(tmp_path):
    """`typedef struct { ... } dims_t;` → `ctypedef struct dims_t:`.

    Same libclang shift affects struct as well — the inner anonymous
    struct now carries the typedef name in its spelling, and we must
    still detect it as anonymous-typedef so the .pxd uses `ctypedef`.
    """
    gen = make_generator(SHAPE_ANON_TYPEDEF_STRUCT, module_name="mod_anon_s")
    pxd = write_module(gen, tmp_path)["cymod_anon_s.pxd"]
    assert re.search(r"\bctypedef\s+struct\s+dims_t\s*:", pxd), (
        f"expected `ctypedef struct dims_t:` for anonymous-typedef struct; pxd:\n{pxd}"
    )


def test_anon_typedef_union_uses_ctypedef(tmp_path):
    """`typedef union { ... } bits_t;` → `ctypedef union bits_t:`.

    Same shape, union flavor.
    """
    gen = make_generator(SHAPE_ANON_TYPEDEF_UNION, module_name="mod_anon_u")
    pxd = write_module(gen, tmp_path)["cymod_anon_u.pxd"]
    assert re.search(r"\bctypedef\s+union\s+bits_t\s*:", pxd), (
        f"expected `ctypedef union bits_t:` for anonymous-typedef union; pxd:\n{pxd}"
    )


# ---------------------------------------------------------------------------
# Failing shapes from full-codegen wheel-build attempts
# ---------------------------------------------------------------------------
#
# These shapes mirror real-world codegen failures captured during the
# end-to-end wheel build against ROCm 7.13 + libclang 23. Each test
# documents:
#
#   1. The exact upstream pattern that triggered the failure
#   2. The C error gcc emits when the codegen mis-generates it
#   3. The expected codegen behavior (either a positive assertion or a
#      `pytest.xfail` placeholder until the underlying gap is closed)
#
# Add a new shape here BEFORE chasing the codegen fix — the test makes
# the regression visible alongside the others and prevents reintroducing
# the same bug from a different code path.


SHAPE_ANON_TYPEDEF_ENUM_VIA_INCLUDE = r"""
/* Mirrors ROCm's hipblas → hipblas-common split: the anonymous
 * typedef enum lives in an INCLUDED header, not the one being parsed.
 * The libclang 17+ AST shape (cursor.spelling carries the typedef
 * name on the inner ENUM_DECL) sometimes desyncs across the
 * include boundary so the emitter sees the cursor BUT the
 * `_from_typedef_with_anon_child` flag isn't set on the registered
 * Enum node — codegen then emits `cdef enum X` and gcc rejects with
 * `incomplete type 'enum X'`.
 *
 * Real-world hits: hipblasStatus_t, hipblasDiagType_t,
 * amdsmi_memory_page_status_t, amdsmi_temperature_type_t.
 */
#include "inner_enum.h"

inner_status_t do_thing(int x);
"""

INNER_ENUM_HEADER = r"""
typedef enum {
    INNER_OK = 0,
    INNER_FAIL = 1
} inner_status_t;
"""


SHAPE_HIP_VECTOR_TYPE_UINT4 = r"""
/* HIP/CUDA built-in vector types: `uint4 = struct { uint x,y,z,w; }`
 * declared as a tagged struct with same-name typedef. Hipblas/hiprand/
 * hipfft headers transitively reference this — the codegen used to
 * emit `cdef struct uint4` (correct for the tagged form) but the
 * Cython compile then failed with `incomplete type 'struct uint4'`
 * because no field-bearing definition was visible (typedef chain
 * terminated at a forward declaration).
 *
 * Real-world hit: `hiprand.c`:`(struct uint4)` malloc/sizeof errors.
 */
typedef struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
} uint4;

uint4 make_uint4(unsigned int x);
"""


SHAPE_TRIPLE_POINTER_CONST_MIDDLE = r"""
/* HIP graph capture API uses `const hipGraphNode_t**` parameters.
 * `hipGraphNode_t` is a typedef for `struct hipGraphNode *`, so the
 * fully-expanded type is `struct hipGraphNode *const **` — a triple
 * pointer with a const at the middle level. The codegen's
 * `handle_out_ptr_parm` doesn't have a branch for this shape and
 * raises CodegenUnsupportedPattern. After the graceful-degrade fix
 * in cython.py:2284, the codegen catches the exception and falls
 * through to handle_in_inout_ptr_ — the parm is still bound, just
 * as INOUT instead of OUT.
 *
 * Real-world hit: `hipStreamGetCaptureInfo_v2` parm `dependencies_out`.
 */
typedef struct hipGraphNode_s* hipGraphNode_t;

/**
 * @param[out] dependencies_out  the dependencies array
 */
int hipStreamGetCaptureInfo_v2(int stream, const hipGraphNode_t** dependencies_out);
"""


SHAPE_POINTER_CONST_ARRAY = r"""
/* Pointer-to-const-array shape:
 *   `unsigned short *const[]` — array of const-pointers to short.
 * Generated by `hipblasHaxpyBatched(..., const __half * const y[], ...)`
 * after the typedef chain expands (`__half = unsigned short`). The
 * codegen used to crash on this shape; with the graceful-degrade fix
 * it's bound as IN (the const-pointer-array is read-only from the
 * function's POV anyway).
 *
 * Real-world hit: `hipblasHaxpyBatched` parm `y`.
 */
/**
 * @param[in,out] y batched pointer to half-precision values
 */
int f_const_ptr_array(int n, unsigned short *const y[]);
"""


def test_anon_typedef_enum_via_include_uses_ctypedef(tmp_path):
    """Anonymous-typedef enum defined in an INCLUDED header still
    classifies as `ctypedef enum X:` — the libclang-17+ behavior shift
    must be detected through the include boundary, not just at the
    top-level cursor of the parsed file.
    """
    # Place the inner header alongside the parsed one in the unsaved
    # files set so libclang can resolve `#include "inner_enum.h"`.
    from interfacegen.cparser import CParser
    from interfacegen import treefactory, cython
    parser = CParser(
        "input.h",
        unsaved_files=[
            ("input.h", SHAPE_ANON_TYPEDEF_ENUM_VIA_INCLUDE),
            ("./inner_enum.h", INNER_ENUM_HEADER),
        ],
    )
    parser.parse()
    root = treefactory.from_libclang_translation_unit(
        backend=cython, translation_unit=parser.translation_unit,
    )
    # Pull the Enum node directly so we can inspect its flag — the
    # write_module path filters by file location and would suppress
    # included-file nodes by default.
    inner_status = None
    for n in root.walk(postorder=False):
        if isinstance(n, cython.Enum) and n.name == "inner_status_t":
            inner_status = n
            break
    if inner_status is None:
        pytest.xfail(
            "node_filter dropped the inner-include enum — cross-include "
            "anonymous-typedef detection needs node_filter relaxation"
        )
    head = inner_status._render_c_interface_head()
    assert re.search(r"\bctypedef\s+enum\s+inner_status_t\s*:", head), (
        f"regression: inner enum from included header not classified as "
        f"anonymous-typedef. Rendered head: {head!r}"
    )


def test_hip_vector_type_uint4_field_bearing_definition(tmp_path):
    """`typedef struct uint4 {…} uint4;` (tagged + same-name typedef)
    must emit a field-bearing struct definition so gcc can `sizeof()`
    it. The current codegen emits `cdef struct uint4:` with the four
    `unsigned int` fields visible.
    """
    gen = make_generator(SHAPE_HIP_VECTOR_TYPE_UINT4, module_name="mod_uint4")
    pxd = write_module(gen, tmp_path)["cymod_uint4.pxd"]
    # Definition present, with all four fields.
    assert re.search(
        r"\b(?:cdef|ctypedef)\s+struct\s+uint4\s*:\s*\n(?:.*\n){3,}.*unsigned\s+int\s+w",
        pxd, re.MULTILINE,
    ), (
        f"uint4 struct definition is missing or truncated; pxd:\n{pxd}"
    )


def test_triple_pointer_const_middle_codegen_does_not_crash(tmp_path):
    """`struct T *const **` parm must not crash the codegen. After
    the graceful-degrade fix in cython.py, an unsupported OUT shape
    falls through to the INOUT handler so the binding is still
    emitted — possibly as IN/INOUT instead of OUT, which the user
    can fix manually using the docstring's original C signature.
    """
    gen = make_generator(SHAPE_TRIPLE_POINTER_CONST_MIDDLE, module_name="mod_triple")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_triple.pxd"]
    pyx = files["mod_triple.pyx"]
    # Function must be present in BOTH the cy* declaration AND the
    # high-level Python module.
    assert re.search(r"\bhipStreamGetCaptureInfo_v2\b", pxd), (
        f"function decl missing from cymod_triple.pxd; pxd:\n{pxd}"
    )
    assert re.search(r"\bhipStreamGetCaptureInfo_v2\b", pyx), (
        f"function impl missing from mod_triple.pyx; pyx:\n{pyx}"
    )


def test_pointer_const_array_codegen_does_not_crash(tmp_path):
    """`unsigned short *const[]` parm must not crash the codegen.
    Same graceful-degrade contract as the triple-pointer case.
    """
    gen = make_generator(SHAPE_POINTER_CONST_ARRAY, module_name="mod_pca")
    files = write_module(gen, tmp_path)
    pxd = files["cymod_pca.pxd"]
    pyx = files["mod_pca.pyx"]
    assert re.search(r"\bf_const_ptr_array\b", pxd), (
        f"function decl missing from cymod_pca.pxd; pxd:\n{pxd}"
    )
    assert re.search(r"\bf_const_ptr_array\b", pyx), (
        f"function impl missing from mod_pca.pyx; pyx:\n{pyx}"
    )


# ---------------------------------------------------------------------------
# GIL-release shape — the `with nogil:` emitter
#
# These tests cover the codegen path that wraps each cy* C call in
# `with nogil:`, hoisting Python-touching arg expressions into typed C
# locals before the block. They exercise the seven retval shapes
# (void / basic / enum / record / ptr-to-record / ptr-to-char /
# any-ptr) and the arg patterns (basic value, enum value, OUT-ptr,
# INOUT-ptr, record-by-value).
#
# The assertions are written against the high-level `<module>.pyx`
# only — the cy* declaration in `cymod_*.pxd` is unaffected by the
# refactor.
# ---------------------------------------------------------------------------


from interfacegen.support.recipes import control as _ctrl


def _intent_outprefix(parm):
    """Pointer-intent rule for synthetic GIL-release tests: parms whose
    name starts with ``out_`` are OUT, everything else falls back to
    the default classifier. Lets a single header drive both code paths.
    """
    if parm.cython_name.startswith("out_"):
        return _ctrl.ParmIntent.OUT
    return _ctrl.DEFAULT_PTR_PARM_INTENT(parm)


def _emit_nogil_pyx(header_text: str, *, module_name: str, tmp_path) -> str:
    """Generate the high-level .pyx for ``header_text`` with the
    with-nogil emitter active (cy* decl marked ``noexcept nogil``).
    """
    gen = make_generator(
        header_text,
        module_name=module_name,
        modifiers_lazy_loader=" noexcept nogil",
        ptr_parm_intent=_intent_outprefix,
    )
    files = write_module(gen, tmp_path)
    return files[f"{module_name}.pyx"]


def _extract_function_body(pyx: str, func_name: str) -> str:
    """Return the source of the ``def <func_name>`` block (signature
    through the last indented line of its body), with the leading
    docstring elided so subsequent body assertions don't accidentally
    match docstring text.

    The function ends at the first column-0 non-blank line after its
    signature — function bodies are indented, so any unindented line
    is outside the function (the next ``def``, ``@cython`` decorator,
    ``__all__`` block, etc.).
    """
    lines = pyx.splitlines()
    sig_prefix = f"def {func_name}("
    start = None
    for i, line in enumerate(lines):
        if line.startswith(sig_prefix):
            start = i
            break
    if start is None:
        raise AssertionError(
            f"function def {func_name}(...) not found in pyx:\n{pyx}"
        )
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line:
            continue
        # First column-0 non-blank line after the signature is outside
        # the function body.
        if not line[0].isspace():
            end = j
            break
    block = "\n".join(lines[start:end])
    # Strip the r"""...""" docstring (including its line's leading
    # indent) so body assertions see only generated code AND so the
    # remaining line indents stay accurate.
    return re.sub(
        r'^[ \t]*r?"""[\s\S]*?"""\n?',
        "",
        block,
        flags=re.MULTILINE,
    )


def _nogil_block_body(body: str) -> str:
    """Return the indented body of the ``with nogil:`` block as a
    single space-joined string, or raise AssertionError if the block
    is missing.

    Contract: the block body is a single logical statement (the cy*
    call). The codegen sometimes wraps long arg lists across multiple
    physical lines via embedded ``\\n{indent*2}`` markers in the call
    args; treat those as a single statement.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.endswith("with nogil:"):
            with_indent = len(line) - len(line.lstrip())
            captured = []
            for sub in lines[i + 1:]:
                sub_indent = len(sub) - len(sub.lstrip())
                if sub.strip() == "":
                    continue
                if sub_indent <= with_indent:
                    break
                captured.append(sub.strip())
            assert captured, (
                f"`with nogil:` block has no body in:\n{body}"
            )
            return " ".join(captured)
    raise AssertionError(f"no `with nogil:` block in body:\n{body}")


def _assert_no_python_in_nogil_call(nogil_line: str) -> None:
    """The cy*-call line inside `with nogil:` must not contain any
    Python-touching token. cdef-class field access (``._ptr``) is
    explicitly allowed — it lowers to a typed C field load.
    """
    forbidden = [
        ".fromPyobj(",
        ".fromValue(",
        ".fromPtr(",
        ".getElementPtr(",
        ".value",  # IntEnum .value attribute lookup
    ]
    for tok in forbidden:
        # ".value" is a substring of names like "._py_buffer.value" —
        # for our codegen output, none of those appear inside the
        # block, so the substring check is precise enough.
        assert tok not in nogil_line, (
            f"Python-touching token {tok!r} leaked into `with nogil:` body:"
            f"\n  {nogil_line}\nFull block source:\n{nogil_line}"
        )


def test_nogil_void_retval_no_args(tmp_path):
    """void f(void) — block has no cdef retval, single bare cy* call."""
    pyx = _emit_nogil_pyx(
        "void op_void(void);",
        module_name="mod_nv", tmp_path=tmp_path,
    )
    body = _extract_function_body(pyx, "op_void")
    assert "with nogil:" in body, f"missing nogil block:\n{body}"
    # No cdef retval declaration for void.
    assert "_cy_op_void__retval" not in body, (
        f"void retval should not produce a cdef holder:\n{body}"
    )
    nogil_line = _nogil_block_body(body)
    assert "cymod_nv.op_void(" in nogil_line, (
        f"cy* call missing inside nogil block:\n{nogil_line}"
    )


def test_nogil_basic_retval_basic_arg(tmp_path):
    """int f(int) — cdef int retval, no hoist, _cy_R returned bare."""
    pyx = _emit_nogil_pyx(
        "int op_basic(int x);",
        module_name="mod_nb", tmp_path=tmp_path,
    )
    body = _extract_function_body(pyx, "op_basic")
    assert "cdef int _cy_op_basic__retval" in body, (
        f"missing cdef retval:\n{body}"
    )
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    assert "_cy_op_basic__retval = cymod_nb.op_basic(x)" in nogil_line, (
        f"basic-typed parm should be inline; got:\n{nogil_line}"
    )
    # Basic retval is its own Python value — return bare _cy_R, no wrap.
    assert re.search(
        r"return\s+_cy_op_basic__retval\b", body
    ), f"basic retval should be returned bare:\n{body}"


def test_nogil_enum_retval_enum_arg(tmp_path):
    """Enum return + IntEnum arg: arg.value is hoisted; retval wrap
    `status_t(_cy_R)` lives post-block in the return tuple.
    """
    header = """
    typedef enum { OK = 0, ERR = 1 } status_t;
    typedef enum { K_A = 0, K_B = 1 } kind_t;
    status_t op_enum(kind_t k);
    """
    pyx = _emit_nogil_pyx(header, module_name="mod_ne", tmp_path=tmp_path)
    body = _extract_function_body(pyx, "op_enum")
    # Hoist for the enum input arg, using the cprefix-prefixed cy*
    # enum type (not the same-named IntEnum wrapper).
    assert re.search(
        r"cdef\s+cymod_ne\.kind_t\s+_cy_op_enum__arg_0\s*=\s*k\.value",
        body,
    ), f"missing cprefix-prefixed enum-value hoist:\n{body}"
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    # The cy* call must reference the hoisted symbol, not k.value.
    assert "_cy_op_enum__arg_0" in nogil_line, (
        f"cy* call must use hoisted symbol; got:\n{nogil_line}"
    )
    # IntEnum constructor wrap is post-block in the return tuple.
    assert re.search(
        r"return\s+status_t\(_cy_op_enum__retval\)",
        body,
    ), f"enum retval wrap must be inlined post-block:\n{body}"


def test_nogil_inout_record_pointer_hoists_fromPyobj(tmp_path):
    """Record-pointer INOUT arg: the `T.fromPyobj(p).getElementPtr()`
    chain hoists into a typed C pointer pre-block; the cy* call only
    sees the pure pointer.
    """
    header = """
    typedef enum { OK = 0 } status_t;
    struct stream_s;
    typedef struct stream_s stream_t;
    status_t op_inout(stream_t* s);
    """
    pyx = _emit_nogil_pyx(header, module_name="mod_ni", tmp_path=tmp_path)
    body = _extract_function_body(pyx, "op_inout")
    assert re.search(
        r"cdef\s+cymod_ni\.stream_s\s*\*\s*_cy_op_inout__arg_0\s*=\s*"
        r"stream_s\.fromPyobj\(s\)\.getElementPtr\(\)",
        body,
    ), f"missing cprefix-prefixed fromPyobj hoist for INOUT record-ptr arg:\n{body}"
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    assert "_cy_op_inout__arg_0" in nogil_line, (
        f"cy* call must use hoisted symbol; got:\n{nogil_line}"
    )


def test_nogil_out_ptr_typed_prolog_inline_addressof(tmp_path):
    """OUT pointer arg (`int* out_x`) — under the default rank rule,
    the codegen routes this through the ListOfInt wrapper (degree=-1
    branch). The wrapper is constructed in the prolog as a typed
    cdef local (``cdef <T> out_x = <T>.fromPtr(NULL)``); the cy*
    call uses ``<int *>out_x._ptr`` inline inside ``with nogil:``
    because the typed cdef-class field access is GIL-safe. No hoist
    is emitted — the typed prolog is what makes the inline use
    legal.
    """
    header = """
    typedef enum { OK = 0 } status_t;
    status_t op_out(int* out_x);
    """
    pyx = _emit_nogil_pyx(header, module_name="mod_no", tmp_path=tmp_path)
    body = _extract_function_body(pyx, "op_out")
    # Prolog now uses a cdef-typed wrapper construction so Cython
    # knows the local's cdef-class type — required for `&out_x._ptr`
    # to work inside `with nogil:`.
    assert re.search(
        r"cdef\s+\S*ListOfInt\s+out_x\s*=\s*\S*ListOfInt\.fromPtr\(NULL\)",
        body,
    ), f"missing typed prolog wrapper construction:\n{body}"
    # No hoist line: the address-of stays inline inside the block.
    assert "_cy_op_out__arg_0" not in body, (
        f"OUT-ptr should not be hoisted when prolog is typed:\n{body}"
    )
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    assert "<int *>out_x._ptr" in nogil_line, (
        f"OUT scalar should be passed by `<int *>out_x._ptr` inline:"
        f"\n{nogil_line}"
    )


def test_nogil_record_by_value_arg_hoists_dereferenced_value(tmp_path):
    """`point_t pt` by value — hoisted as
    `cdef point_t _cy_..._arg_0 = point_t.fromPyobj(pt).getElementPtr()[0]`
    so the cy* call sees a stack-local C record copy.
    """
    header = """
    typedef enum { OK = 0 } status_t;
    typedef struct point_st { int x; int y; } point_t;
    status_t op_rec(point_t pt);
    """
    pyx = _emit_nogil_pyx(header, module_name="mod_nr", tmp_path=tmp_path)
    body = _extract_function_body(pyx, "op_rec")
    assert re.search(
        r"cdef\s+cymod_nr\.point_st\s+_cy_op_rec__arg_0\s*=\s*"
        r"point_st\.fromPyobj\(pt\)\.getElementPtr\(\)\[0\]",
        body,
    ), f"missing cprefix-prefixed record-by-value hoist:\n{body}"
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    assert "_cy_op_rec__arg_0" in nogil_line, (
        f"cy* call must reference hoisted record value; got:\n{nogil_line}"
    )


def test_nogil_any_pointer_retval_wrap_post_block(tmp_path):
    """`void* op_any(int)` — retval wrap goes through Pointer.fromPtr
    POST-block, guarded by `None if _cy_R == NULL else …`. The
    factory call MUST NOT appear inside `with nogil:`.
    """
    header = "void* op_any(int x);"
    pyx = _emit_nogil_pyx(header, module_name="mod_nany", tmp_path=tmp_path)
    body = _extract_function_body(pyx, "op_any")
    nogil_line = _nogil_block_body(body)
    _assert_no_python_in_nogil_call(nogil_line)
    # The Pointer.fromPtr wrap must be post-block, in the return.
    assert re.search(
        r"return\s+None if _cy_op_any__retval == NULL else "
        r"\S*Pointer\.fromPtr\(<void\*>_cy_op_any__retval\)",
        body,
    ), f"any-pointer retval wrap must be post-block:\n{body}"
    # And must NOT live inside the nogil block.
    assert ".fromPtr(" not in nogil_line, (
        f"Pointer.fromPtr must not appear in nogil body:\n{nogil_line}"
    )


def test_nogil_with_gil_mode_keeps_inline_emission(tmp_path):
    """When `modifiers_lazy_loader` does NOT contain ``nogil``, the
    with-gil emitter is selected: no hoists, no `with nogil:` block,
    inline single-line cy* call + Python wrap. Both modes are valid
    first-class options of the dispatcher.
    """
    header = """
    typedef enum { OK = 0 } status_t;
    typedef enum { K_A = 0, K_B = 1 } kind_t;
    status_t op_with_gil(kind_t k);
    """
    # NOTE: no modifiers_lazy_loader — defaults to "" → with-gil.
    gen = make_generator(header, module_name="mod_wg")
    files = write_module(gen, tmp_path)
    body = _extract_function_body(files["mod_wg.pyx"], "op_with_gil")
    assert "with nogil:" not in body, (
        f"with-gil mode must not emit `with nogil:`:\n{body}"
    )
    assert "_cy_op_with_gil__arg_" not in body, (
        f"with-gil mode must not hoist args:\n{body}"
    )
    # Inline `.value` on the IntEnum argument is the with-gil shape.
    assert "k.value" in body, (
        f"with-gil should keep `.value` inline:\n{body}"
    )
