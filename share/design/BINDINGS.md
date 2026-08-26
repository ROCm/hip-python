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


## Return-value contract: status-first tuple

Every non-LLVM library binding returns a `tuple` whose first
element is the library's status code/enum (`hipError_t`,
`hiprtcResult`, `amd_comgr_status_t`, `hipblasStatus_t`,
`hsa_status_t`, ...), followed by any OUT values. Callers can
therefore always unpack uniformly:

```python
status, *out_values = some_binding(...)
```

This is driven by the per-generator
`module_opts["python_interface_always_return_tuple"]` flag (a key
on `CythonModuleGenerator`, default `False`). The flag interacts
with two emission paths in
`interfacegen.cython.Function.render_python_interface_impl`:

| Case | What the generator emits |
|---|---|
| C function returns the status enum | The status wrap (`hipError_t(_cy_..._retval)`) is `out_args.insert(0, ...)`'d as element 0 automatically. |
| C function returns a non-status value (`const char*`, `int`, `void`, a *different* enum) | The recipe's `node_init` prepends a synthetic success code via `prepend_python_return_value(...)` (e.g. `hipError_t.hipSuccess`, `hiprtcResult.HIPRTC_SUCCESS`, `hipblasStatus_t.HIPBLAS_STATUS_SUCCESS`), so element 0 is still a status. |
| Exactly one return value total | With the flag set, a 1-tuple `(status,)` is emitted instead of a bare `status`, so the shape never collapses. |

Concretely, the prepend lives in each per-library generator's
`node_init`: `generators_hip.py`'s `hip_node_init` /
`hiprtc_node_init`, and the shared `_make_status_node_init(prefix,
status_type, success_const)` helper in `generators_libraries.py`
and `generators_systems.py`. The same helper also downgrades the
function's lazy-loader modifier from `except? <SENTINEL> nogil` to
`noexcept nogil` for these non-status returns (the `except?`
sentinel only type-checks when the function actually returns the
status enum).

Two systems modules (`roctx`, `hipfile`) have no status enum at
all — they set the flag purely for shape uniformity (bare returns
become 1-tuples) and prepend nothing.

The LLVM C bindings opt **out** (`module_opts={"python_interface
_always_return_tuple": False}` in `generators_compiler.py`'s
`write_llvm_modules`): the LLVM-C API has no status-return
convention, so its wrappers return bare values, matching upstream.


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

Hoisting is not only a GIL device, which is why the table above holds
for the with-gil emitter too. Where the hoisted expression borrows a
pointer from an adapter — the `ListOfBytes` / `Pointer` / record-wrapper
rows — inlining the chain into the call expression is a
use-after-free regardless of the GIL: Cython drops an intermediate
object as soon as the object itself is no longer needed, which is the
moment `getPtr()` returns, so the generated C decrefs the adapter on
the line *before* the call and `__dealloc__` frees the array the callee
is about to read. That is why such an argument is rendered as two
locals (`_cy_..._arg_N_obj` holding the adapter, `_cy_..._arg_N`
holding the pointer) and why
`interfacegen.cython.CallArgHoist.render_prehoist` is the only
rendering. The LLVM bindings, the only ones the with-gil emitter
produces, shipped exactly this bug: a freed `void *[2]` reached
`LLVMFunctionType` after glibc had overwritten both slots with tcache
bookkeeping, and the first `LLVMGetParam` on the resulting type
segfaulted. Coverage lives in
`test_codegen_wrapper_arg_lifetime.py`.

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
falls back to the with-gil emitter — both modes are first-class
options. That emitter drops the `with nogil:` block and keeps the
retval wrap inline in the call expression, but the argument hoists
are identical: the hoisting rule is emitter-independent (see
below).


## Loader error contract — the implicit `except 1` return

The handcoded loaders in `rocm-bindings-core`
(`rocm/bindings/util/posixloader.pyx` and `win32loader.pyx`) follow a
contract that is easy to misread, because the error return value is
*never written in the source*:

```cython
cdef int open_library(void** lib_handle, const char* path) except 1 nogil:
    lib_handle[0] = posix.dlfcn.dlopen(path, posix.dlfcn.RTLD_NOW)
    cdef char* reason = NULL
    if lib_handle[0] == NULL:
        reason = posix.dlfcn.dlerror()
        with gil:
            raise RuntimeError(f"failed to dlopen '{str(path)}': {str(reason)}")
    return 0
```

`open_library`, `close_library`, and `load_symbol` are all declared
`cdef int ... except 1 nogil`. The success path is an explicit
`return 0`; the error path is a `raise` inside a `with gil:` block.
There is no `return 1` anywhere — and there must not be.

**The `1` is synthesized by Cython.** For an `except 1` function
(note: no `?`, so `1` is an unambiguous error sentinel, never a
legitimate return value), Cython generates a C return value of `1`
whenever an exception propagates out of the body. Writing `return 1`
by hand would be redundant at best and would defeat the sentinel at
worst. The docstrings' "Positive number if something has gone wrong,
'0' otherwise" describe this Cython-synthesized value, not literal
source code.

**Why `with gil:` is mandatory.** A `raise` touches the Python
runtime (it allocates the exception object and sets the thread-state
error indicator), which is illegal inside a `nogil` body without
first reacquiring the GIL. The `with gil:` block does two things at
once: it sets the Python error indicator (`PyErr`) *and* it triggers
the `except 1` error path that produces the `1` return value. Both
effects are required — the return value tells a `nogil` caller that
something failed without it having to touch Python; the error
indicator carries the actual exception for whenever the GIL is next
held.

**Caller side.** The generated `__init` / `__init_symbol` helpers in
`interfacegen.cython._backend` are themselves `except 1 nogil` and
chain the contract upward:

```cython
cdef int __init() except 1 nogil:
    ...
        return loader.open_library(&_lib_handle, dll)   # propagates 0 or 1
    return 0

cdef int __init_symbol(void** result, const char* name) except 1 nogil:
    ...
        init_result = __init()
        if init_result > 0:        # non-zero ⇒ open_library failed
            return init_result
    ...
        return loader.load_symbol(result, _lib_handle, name)
    return 0
```

Because every layer shares the same `except 1` sentinel, a single
`raise` deep in `open_library` has two simultaneous consequences:

1. the non-zero `1` return propagates up the `cdef` call chain (each
   `except 1` caller sees the sentinel and re-enters its own error
   path), and
2. the Python error indicator stays set, so the original
   `RuntimeError` re-raises as an ordinary Python exception the moment
   control reaches a GIL-holding Python entry point (the high-level
   `def` wrapper or `__init_symbol`'s callers).

A caller that only ever inspects the `int` return value will still
behave correctly — it sees a non-zero value and bails — but the
exception is *not* lost; it surfaces with its full message once the
GIL is reacquired.

**Contrast with `has_symbol`.** The non-raising probe
`has_symbol(...)` is declared `noexcept nogil` and returns a real
`bint`. It must never raise, so it has no error sentinel to reserve.
This is exactly why the generated `__has_symbol` cannot simply call
`__init()` and let an exception propagate — it wraps the call in
`with gil: try/except` and returns `False` on failure instead of
threading an error value through:

```cython
cdef bint __has_symbol(const char* name) noexcept nogil:
    ...
    if _lib_handle == NULL:
        with gil:
            try:
                init_result = __init()
            except Exception:
                return False
        ...
```

**Cross-platform parity.** The posix loader (`dlopen`/`dlsym`/
`dlclose`) and the win32 loader (`LoadLibraryA`/`GetProcAddress`/
`FreeLibrary`) implement the identical `except 1` contract — only the
underlying OS calls differ. Both are handcoded and committed in
`rocm-bindings-core` as `src/rocm/bindings/util/posixloader.pyx` and
`win32loader.pyx`, mirroring the same contract.


## Naming convention for generated locals

All generated locals follow a uniform `_cy_<func>__<role>` prefix
so they don't clash with user-visible parm names or recipe
prolog/epilog locals:

| Symbol | Role | Lifetime |
|---|---|---|
| `_cy_<func>__retval` | C-level return-value holder for the cy* call. Declared `cdef <C-retval-type>` before `with nogil:`, assigned inside it. | Whole function body |
| `_cy_<func>__arg_<N>` | Hoisted typed C local for a Python-touching arg expression. Declared `cdef <hoist-type> _cy_..._arg_N = <expr>` immediately before the cy* call, in both emitters; the call references the symbol by name. Only emitted for Python-touching args; pure-C args stay inline. | Whole function body |
| `_cy_<func>__arg_<N>_obj` | Companion local for a *wrapper-bound* hoist: the adapter instance (`ListOfBytes`, `Pointer`, a record wrapper) that owns the memory `_cy_..._arg_N` points into. The pointer is extracted from it on the next line, so the adapter is still referenced when the C call runs. | Whole function body |
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

See [CODEGEN.md](CODEGEN.md) for where intent rules slot into the
overall pipeline, and [POINTER_ARGUMENTS.md](POINTER_ARGUMENTS.md) for
the pointer intent/degree decision table; the rule chain itself lives in
`tools/interfacegen/python/interfacegen/support/recipes/`.


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
  the `version.py.in` template — see CODEGEN.md §"Forbidden outputs".
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
