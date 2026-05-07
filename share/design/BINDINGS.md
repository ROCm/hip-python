# Anatomy of the generated HIP Python bindings

This document describes the *runtime contract* of the generated
bindings — what shape the `interfacegen` code generator emits and
why each part is shaped the way it is. It is the companion to
[CODEGEN.md](CODEGEN.md), which describes the *generation pipeline*
(repos, branches, generator-owned vs. handcoded outputs).

The implementation insights captured here are sourced from the cy*
call-site `with nogil:` refactor — the moment the high-level
wrappers grew enough internal structure that a single-page reference
became worthwhile.


## Two-tier wrapper layout

Every ROCm library produces two Cython modules:

| File | Role |
|---|---|
| `cy<lib>.pxd` / `cy<lib>.pyx` | C-level wrapper. `cdef`-only declarations, `noexcept nogil` modifiers, lazy-loader bodies that `dlsym` the underlying `libhip<lib>.so` symbols. Consumed via `cimport` from other Cython code. No Python-level entry points. |
| `<lib>.pxd` / `<lib>.pyx` | High-level wrapper. Python-callable `def` functions and IntEnum classes. Wraps cy* C calls in the Python type system: argument validation, IntEnum coercion, output-handle materialization, error-code wrapping. |

The split exists for two reasons: cy* is the only API a downstream
Cython consumer needs (`cimport` + direct C call, no Python overhead);
the high-level wrapper carries everything the Python-level user
expects (typed exceptions, `None`-on-NULL guards, IntEnum return
values).

The cy* prefix replaced an earlier `c<lib>` prefix to disambiguate
Cython-level wrappers from any C-level identifier with the `c`
prefix — see [CODEGEN.md](CODEGEN.md) §"File-prefix history".


## Anatomy of a generated function

A representative high-level function (`hipStreamCreateWithFlags`)
emits as:

```cython
@cython.embedsignature(True)
def hipStreamCreateWithFlags(unsigned int flags):
    """..."""
    stream = ihipStream_t.fromPtr(NULL)              # ① prolog
    cdef cyhip.hipError_t _cy_hipStreamCreateWithFlags__retval   # ② cdef retval
    with nogil:                                       # ③ nogil block
        _cy_hipStreamCreateWithFlags__retval = cyhip.hipStreamCreateWithFlags(
            <cyhip.ihipStream_t**>&stream._ptr,
            flags,
        )
    return (                                          # ④ post-block return
        hipError_t(_cy_hipStreamCreateWithFlags__retval),
        None if stream._ptr == NULL else stream,
    )
```

Four regions:

1. **Prolog.** OUT-arg wrapper construction (`stream = ihipStream_t
   .fromPtr(NULL)`), IntEnum input-arg validation (`if not isinstance
   (kind, _kind_t__Base): raise TypeError(...)`), and any user-
   prescribed prolog from the recipe. Runs with the GIL held.

2. **`cdef` retval holder.** A C-level local that will receive the
   cy* function's raw return value. Skipped for void-returning
   functions. Named `_cy_<func>__retval`.

3. **`with nogil:` block.** Single-statement body: the cy* C call.
   Argument expressions inside the block are pure C — typed parm
   references, cdef-class field accesses (`stream._ptr`), address-of
   operators, constantarray casts. Anything that touches the Python
   runtime is hoisted out (see GIL semantics below).

4. **Return tuple.** Python-side wrapping happens here, after the
   GIL is reacquired: IntEnum constructor for enum returns, `T
   .fromValue(...)` for record returns, `T.fromPtr(...)` (with
   `None if ... == NULL else ...` guards) for pointer returns. OUT-arg
   wrappers are returned alongside the wrapped retval.

The split lets ROCm runtime calls that block the host thread
(`hipDeviceSynchronize`, `hipMemcpyAsync`, kernel launches) drop
the GIL during the wait, so other Python threads in the process
can make progress. This was the trigger for the current emission
shape; before the refactor the cy* call was inlined inside a
single Python expression and held the GIL throughout.


## GIL semantics — what runs where

The high-level wrapper alternates between GIL-held and
GIL-released regions in a fixed pattern:

| Region | GIL held? | What runs there |
|---|---|---|
| Prolog (region ①) | YES | OUT-arg `fromPtr(NULL)` initializers, IntEnum validation, recipe prolog |
| Hoists (between ① and ③) | YES | `cdef T _cy_<func>__arg_N = <expr>` lines for Python-touching arg expressions |
| `cdef` retval (region ②) | YES | Just a declaration; no executable Python |
| `with nogil:` body (region ③) | NO | The cy* C call + pure-C arg expressions only |
| Return tuple (region ④) | YES | `IntEnum(...)`, `T.fromValue(...)`, `T.fromPtr(...)`, `None if ... else ...` guards |

What "touches Python" means for an argument expression:

| Expression | Touches Python? | Where it lives |
|---|---|---|
| `flags` (typed `cdef unsigned int` parm) | NO — pure C | Inline inside `with nogil:` |
| `<cyhip.T**>&stream._ptr` (cdef-class field, address-of) | NO — typed C field load | Inline inside `with nogil:` |
| `<cast>&parm._ptr` on a wrapper from prolog | NO — same as above (the prolog assignment infers the cdef-class type) | Inline inside `with nogil:` |
| `parm.value` on an IntEnum | YES — Python attribute lookup | Hoisted to `cdef <enum_c_type> _cy_..._arg_N = parm.value` |
| `T.fromPyobj(parm).getElementPtr()` | YES — cdef staticmethod call (not nogil-callable) | Hoisted to `cdef <T*> _cy_..._arg_N = <expr>` |
| `T.fromPyobj(parm).getElementPtr()[0]` (record by value) | YES — same as above | Hoisted to `cdef <T> _cy_..._arg_N = <expr>` (record copied by value) |
| `<C-type>handler.fromPyobj(parm)._ptr` (datahandle) | YES — `.fromPyobj` is cdef staticmethod | Hoisted to `cdef <C-type> _cy_..._arg_N = <cast><expr>` |

The retval wrap (`hipError_t(...)`, `T.fromValue(...)`, `T.fromPtr
(...)`) goes through Python's type machinery (`__call__`, `__new__`,
`__init__`) and therefore cannot live inside `with nogil:`. It is
inlined directly into the return tuple — there is no
`_py_<func>__retval` intermediate; the wrap appears once and only
feeds the return tuple.

The whole arrangement assumes the cy* declaration is `noexcept
nogil` (or at least `nogil`). Every per-library generator
(`generators_hip.py`, `generators_libraries.py`,
`generators_systems.py`, `generators_compiler.py`,
`cuda_interop.py`) sets `modifiers_lazy_loader` accordingly. If a
recipe ever omits `nogil`, the dispatcher in
`interfacegen.cython.Function._render_python_interface_c_interface_call`
falls back to the with-gil emitter (single-line cy* call + inline
Python wrap) — both modes are first-class options.


## Naming convention for generated locals

All generated locals follow a uniform `_cy_<func>__<role>` prefix
so they don't clash with user-visible parm names or recipe
prolog/epilog locals:

| Symbol | Role | Lifetime |
|---|---|---|
| `_cy_<func>__retval` | C-level return-value holder for the cy* call. Declared `cdef <C-retval-type>` before `with nogil:`, assigned inside it. | Whole function body |
| `_cy_<func>__arg_<N>` | Hoisted typed C local for a Python-touching arg expression. Declared `cdef <hoist-type> _cy_..._arg_N = <expr>` immediately before the cy* call. The cy*-call inside `with nogil:` references the symbol by name. Only emitted for Python-touching args; pure-C args stay inline. | Whole function body |
| `_<func>__retval` | The with-gil emitter's retval name (legacy from before the refactor — preserved by the with-gil branch only). | Whole function body |

Why no `_py_<func>__retval`: the Python wrap of the C-level retval
appears exactly once and only feeds the return tuple. Inlining it
directly avoids a redundant local and matches the with-gil
emitter's shape (which has always inlined the wrap inside the call
expression).


## Pointer-argument intent classification

The codegen has to choose, for each pointer parameter, whether it
is IN, OUT, or INOUT. That choice picks one of the handler shapes
in
`interfacegen.cython.Function._analyze_parms` (e.g.
`handle_out_ptr_parm` for OUT,
`handle_in_inout_ptr_` for IN/INOUT) and therefore drives the
arg-expression shape that ends up in the cy* call.

Intent is decided by a chain of rules attached to the recipe:

1. **Per-library hardcoded rules** (highest precedence). E.g.
   `recipes/rocm.py:rocblas.ptr_parm_intent` knows that
   `rocblas_get_pointer_mode`'s `mode` parm is OUT.
2. **`documented_param_intent`** (generic). Parses Doxygen
   `@param[in|out|in,out]` and `\param[...]` annotations from the C
   declaration's preceding comment block.
3. **`generic.conservative.ptr_parm_intent`**. Treats single
   pointers as IN by default; double pointers as INOUT.
4. **`control.DEFAULT_PTR_PARM_INTENT`**. Final fallback —
   double-pointer-to-non-const → INOUT, otherwise unclassified
   (the IN/INOUT handler runs).

Once intent is decided, the handler shape determines whether the
cy* call sees:

- A pure-C `&out_x` (OUT scalar) or `<cast>&parm._ptr` (OUT
  wrapper) — inline inside `with nogil:`, no hoist.
- A hoisted `_cy_<func>__arg_N` symbol referencing a typed C local
  that captured the result of `T.fromPyobj(parm).getElementPtr()`
  (IN/INOUT wrapper).

See `share/design/CODEGEN.md` for where intent rules slot into the
overall pipeline; the rule chain itself lives in
`interfacegen/python/interfacegen/support/recipes/`.


## Handcoded helpers the generator builds on

The generator emits Cython code that depends on a small handcoded
runtime in `rocm-bindings-core` (path:
`packages/rocm-bindings-core/src/rocm/bindings/util/types.pxd` and
`types.pyx`). Three primitives matter for understanding the
generated output:

| Primitive | What it provides | Why generated code can use it inside `with nogil:` |
|---|---|---|
| `cdef class Pointer { cdef void* _ptr; ... }` | The base wrapper for any opaque pointer the C API returns. | `_ptr` is a typed C field on a cdef class. When a wrapper is constructed in the prolog (`stream = ihipStream_t.fromPtr(NULL)`), Cython infers the local's type from the cdef-staticmethod return signature. Field access then lowers to a typed C load — GIL-safe inside `with nogil:`. |
| `@staticmethod cdef T fromPtr(void* ptr)` / `cdef T fromPyobj(object pyobj)` | Factory methods that wrap a raw pointer or coerce a Python object into a typed wrapper. | These ARE NOT `nogil`-callable (they may allocate or touch Python). They run during the prolog (with GIL held); their return value gets stored either in the wrapper local or hoisted into a `_cy_<func>__arg_N` C local. |
| `cdef T* getElementPtr(self)` (per-wrapper) | Returns the typed C pointer that the wrapper holds. | Also NOT `nogil`-callable, but the codegen never calls it inside `with nogil:` — it appears only in hoist RHS expressions, which run pre-block with GIL held. |

Without these primitives, the GIL-release emission shape would not
be possible: every Python-typed argument would have to be coerced
inside the nogil block and immediately fail.


## What is NOT generator-controlled

Everything described above is what the generator *emits*. The
generator does not own:

- `__init__.py` files, `pyproject.toml`, `setup.cfg`, `MANIFEST.in`,
  `_version.py.in` templates — see CODEGEN.md §"Forbidden outputs".
- The `rocm-bindings-core` runtime (loader, types, paths) and the
  `rocm-bindings-hip` helper modules (`_hip_helpers`,
  `_hiprtc_helpers`).
- Per-package `CMakeLists.txt` and the build infrastructure in
  `cmake/HipPythonBuild.cmake`.

A bare clone of the codegen base branch is not buildable — the
generator must run first to populate the per-library `.pxd`/`.pyx`
files and `cmake/generated_*.cmake` lists. See CODEGEN.md
§"Producing a new release" for the end-to-end flow.


## See also

- [CODEGEN.md](CODEGEN.md) — the code-generation pipeline (repos,
  branches, generator-owned vs. handcoded outputs).
- [BUILDING.md](BUILDING.md) — the hip-python build system in
  detail.
