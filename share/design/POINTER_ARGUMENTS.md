# Pointer-argument intent and rank in interfacegen

## 1. Motivation

`interfacegen` produces Python/Cython/Fortran bindings from C headers via
libclang. Every pointer parameter in a C function carries two attributes the
Python binding needs to know:

- **intent** — does the function read, write, or both read and write the data
  the pointer refers to?
- **rank** — does the pointer refer to a single element (scalar) or to a
  contiguous run of elements (array)?

C does not say either of these out loud. `int *p` could be a scalar OUT
slot, a 1-D buffer the function reads, or both. If the binding generator
guesses wrong it produces an incorrect or unsafe Python signature: a buffer
you must pre-allocate looks like a Python `int` you receive as a return
value, or a read-only buffer is wrapped in a way that lets the user mutate
shared memory.

The codegen needs a way to (a) deduce what C alone proves, (b) admit when
C alone doesn't decide, and (c) layer library-specific knowledge on top.
This document describes the architecture for that.

## 2. What modifiers tell us

A C parameter declaration carries up to three signals about the
pointed-to data: pointer degree, the kind of the innermost layer, and
`const` placement.

| C declaration              | Pointer degree | Innermost layer kind     | Pointee `const` | Rank verdict   | Intent verdict |
|----------------------------|---------------:|--------------------------|-----------------|----------------|----------------|
| `T x` (by value)           |              0 | basic/record/enum        | n/a             | scalar         | n/a (not a pointer parm) |
| `const T x` (by value)     |              0 | basic/record/enum        | yes             | scalar         | n/a |
| `T *p`                     |              1 | POINTER → T              | no              | **unknown**    | **unknown** |
| `const T *p`               |              1 | POINTER → const T        | yes             | unknown        | **IN** |
| `T p[]`  (incomplete arr)  |              1 | INCOMPLETEARRAY → T      | no              | **array (≥1)** | unknown |
| `const T p[]`              |              1 | INCOMPLETEARRAY → const T| yes             | **array (≥1)** | **IN** |
| `T p[N]` (constant arr)    |              0 | CONSTANTARRAY → T        | no              | **array (≥1)** | unknown |
| `const T p[N]`             |              0 | CONSTANTARRAY → const T  | yes             | **array (≥1)** | **IN** |
| `T **p`                    |              2 | POINTER → POINTER → T    | no              | unknown        | unknown |
| `const T **p`              |              2 | POINTER → POINTER → const T | yes          | unknown        | **IN** |
| `T *const p`               |              1 | POINTER (const ptr) → T  | no              | unknown        | unknown — ptr-const is not a data-mutability signal |
| `void *p`                  |              1 | POINTER → void           | no              | unknown        | unknown |
| `const void *p`            |              1 | POINTER → const void     | yes             | unknown        | **IN** |
| `T (*fn)(...)` function ptr |             1 | POINTER → FUNCTIONPROTO | n/a             | n/a            | n/a |

This table is the **single source of truth** for "is this deducible from C
alone?". It is implemented verbatim by the `conservative` ruleset in
`interfacegen/support/recipes/generic.py`.

### Rules distilled

**Rank, conservatively decidable from modifiers**

- INCOMPLETEARRAY layer in the type → rank ≥ 1 (return `1`).
- CONSTANTARRAY layer in the type → rank ≥ 1 (return `1`).
- Plain `*T` / `**T` with no array layer → unknown (`None`).
- Function-pointer parm → unknown (`None`); rank doesn't apply.

**Intent, conservatively decidable from modifiers**

- The ultimately-referred data is `const`-qualified (any non-outer-pointer
  layer is `const`) → `ParmIntent.IN`.
- Anything else → unknown (`None`).

**Explicitly NOT decidable from modifiers**

- `T *` non-const: could be IN, OUT, or INOUT. Must come from library
  knowledge or a documented convention.
- `T **` non-const: same — common convention is OUT/INOUT (handle
  creation), but that's convention, not deduction.
- `T *const` (top-level pointer-const): irrelevant to data mutability.
- `restrict`, `volatile`: not intent signals.

## 3. Catalog of conventions

The conservative ruleset is correct but very narrow — it answers only
when C proves the answer. Real C APIs follow widespread conventions that
classify a large fraction of remaining parameters correctly. Each
convention below is its own opt-in ruleset class in `generic.py`. A recipe
author wires them per-library via `@fallback(...)` so the recipe picks
the stack that matches the library family.

| # | Convention                          | Trigger                                                                                          | Verdict                                  | Sourced from |
|--:|-------------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------|--------------|
| 1 | **conservative**                    | innermost-pointee `const`, OR an array layer present                                            | IN; rank ≥ 1 only when an array layer exists | SAL (`_In_` ⇒ read-only), GIR, COM/IDL — universal agreement |
| 2 | **pointer_as_reference**            | non-const `T*` parm                                                                              | INOUT                                    | "Reference school" — Ropert; cprogramming.com |
| 3 | **pointer_as_value**                | non-const `T*` parm whose pointee is not itself a pointer                                       | IN                                       | "Value school" — cprogramming.com |
| 4 | **double_indirection_out**          | non-const `T**` (esp. `void**`, `struct**`, `enum**`)                                           | OUT_CALLEE_ALLOCATED; rank 0 for a typed `T**` handle slot, **rank 1 for `void**`** (untyped callee-allocated byte buffer, the `hipMalloc` idiom). Callee-allocation is the hint, *independent* of rank. | GIR `(out)` for double-indirection on a structure parameter; SAL `_Outptr_`. COM `[out]` requires a pointer. |
| 5 | **string_z**                        | `const char *` ⇒ IN; `char *` ⇒ ambiguous intent; `char **` ⇒ OUT_CALLEE_ALLOCATED. **All char pointers are rank 1** — a NUL-terminated string is rank-1 data (see §4) | as listed | GIR / SAL `_In_z_`, `_Outptr_result_z_` |
| 6 | **array_with_length_param**         | `T *buf` adjacent to integer parm whose name names the length (`n`, `len`, `count`, `*_size`)  | `buf` is rank 1; intent follows pointee const | GIR `(array length=N)`. SAL `_In_reads_(n)` / `_Out_writes_(n)`. *(deferred — relational rule)* |
| 7 | **zero_terminated_array**           | `T**` where elements are sentinel-terminated                                                    | rank 1                                   | GIR `(array zero-terminated=1)`. *(deferred)* |
| 8 | **status_return_out_pointer**       | function return type is an integer/error status enum AND parm is the only non-const pointer parm | that parm is OUT                         | C tradition: status code as return value, results via OUT pointer (POSIX, every Khronos API, every ROCm runtime). *(deferred — needs return-type access)* |
| 9 | **opaque_typedef_is_handle**        | type is a typedef whose canonical form is `T*` **with `T` not `void`**                            | rank 0 (the typedef is a scalar handle). **`void*`-canonical typedefs (`hipDeviceptr_t`, `CUdeviceptr`) are excluded** — a `void*` is untyped byte storage, a buffer, not a scalar handle, so the rule defers and the parm ranks as a rank-1 buffer. | Universal opaque-handle pattern (`FILE*`, `cl_context`, `VkInstance`, `cudaStream_t`, `hipStream_t`) — but **not** `void*` device-pointer aliases |
| 10 | **documented_param_intent**         | parent function's raw doxygen comment carries `@param[in\|out\|in,out] <pname>` (or `\param[…]`) | the documented intent (IN / OUT / INOUT); a `[out]` on a **callee-allocated shape** refines to OUT_CALLEE_ALLOCATED. The shape test is structural, not rank-only: rank-0 scalar/handle/string, OR a non-const double-indirection `T**`/`void**`, OR a `char**`. A `[out]` on a caller-sized buffer (a single `T*` the callee fills) stays plain OUT. | Doxygen direction tags — explicit author intent. Originated as `amdsmi._doxygen_intent`; an audit found it resolved 88 mismatches a verb-based heuristic produced for amdsmi.h alone. |

Conventions #6–8 are sketched in `generic.py` with TODO bodies; they need
relational access (sibling parms, function return type) that the
`tree.Parm` API doesn't yet expose ergonomically. They're slated for a
follow-up.

## 4. Char pointers: zero-terminated strings map to `CStr`

A `char *` is, in the overwhelming majority of C APIs, a NUL-terminated
string (or a caller-allocated string buffer); a `char **` is either an OUT
slot that returns one string or an array-of-strings. interfacegen encodes
this as a deliberate, library-agnostic default so bindings expose
`rocm.bindings.util.types.CStr` rather than an opaque `Pointer` for these.

### 4.1 Rank: a string is rank-1 data, regardless of indirection depth

`string_z` is named for the SAL `_z` ("zero-terminated") family. A
zero-terminated string is **rank-1 data** no matter how many pointer layers
wrap it, so `string_z.ptr_rank` reports `1` for **both** `char *` (degree 1)
and `char **` (degree 2). Rank is deliberately *not* the lever that
separates a single string from an array-of-strings, or IN from OUT.

`DEFAULT_PTR_RANK` (the final fallback in `control.py`) likewise returns `1`
for every pointer — `char *` is **no longer special-cased to rank 0**.
Several recipes already forced `char *` → rank 1 (`roctx`, `comgr`,
`llvm_c`); making it the default just aligns the rest of the tree with them.
The rare genuine single-`char`-by-reference parameter is the exception, and
is handled by an explicit per-parameter **rank-0 override** (see §4.4).

### 4.2 Wrapper choice: pointer degree + intent, decided in the handler

`double_indirection_out` (convention #4) claims any `T**` — including
`char **` — as **rank 0** in the runtime chain, so the wrapper for a char
pointer cannot be chosen from rank alone. It is chosen in
`CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER`
(`interfacegen/cython/_defaults.py`) by **pointer degree + intent**:

| Shape     | Intent                       | Wrapper                                                                 |
|-----------|------------------------------|-------------------------------------------------------------------------|
| `char *`  | any (IN / INOUT / return / field) | `CStr` (the string buffer itself; also `char[]`)                   |
| `char **` | OUT                          | `CStr` (the slot returns a single string)                               |
| `char **` | IN / field / return          | falls through to `Pointer` (recipe overrides to `ListOfBytes` for argv) |

The `char **` OUT check is intent-scoped
(`isinstance(node, tree.Parm) and node.is_out_ptr`) and runs *before* the
rank-based branches precisely because `char **` is rank 0 and would
otherwise never be reached. An IN `char **` (an argv-style array of strings)
is intentionally **not** matched, so it is never clobbered into a single
`CStr`.

### 4.3 Binding-allocated OUT string buffers (`hipDeviceGetName`)

Some functions are *caller-allocated* in C — the caller hands in a `char *`
buffer and a length — but read far more naturally in Python as a function
that simply **returns** the filled string. `hipDeviceGetName(char *name,
int len, …)`, `hipDeviceGetPCIBusId(char *pciBusId, int len, …)`, and
`hipGraphInstantiate(…, char *pLogBuffer, size_t bufferSize)` are the
canonical cases. The binding allocates the buffer for the user, passes it,
and returns it as a `CStr`.

Crucially, this is a **Cython-only ergonomic, not a cross-backend truth**: in
C these buffers are caller-allocated, and a direction-only backend (Fortran)
should keep them as plain caller-allocated `OUT`. So the machinery lives in
the HIP *Cython recipe generator* (`tools/hip-python-generate/.../generators_hip.py`),
**not** in the shared `controls.hip` (`support/recipes/rocm.py`) — mirroring
the `hipMalloc` → `DeviceArray` treatment, which is likewise a Cython-only
override in the same generator. A single map drives both halves:

```python
# generators_hip.py — (func, buffer_parm) -> size_parm
_CSTR_OUT_BUFFERS = {
    ("hipDeviceGetName",     "name"):       "len",
    ("hipDeviceGetPCIBusId", "pciBusId"):   "len",
    ("hipGraphInstantiate",  "pLogBuffer"): "bufferSize",
}
```

The behavior is the composition of three pieces:

1. a Cython-only intent wrapper (`hip_ptr_parm_intent`) returns
   **`OUT_CALLEE_ALLOCATED`** for the mapped buffers and otherwise delegates
   to `controls.hip.ptr_parm_intent`. This routes the parm down the
   callee-allocated path (§8) so it becomes a **return value**, dropping out
   of the Python argument list;
2. the recipe `node_init` prepends `<buf>.malloc(<size>)` before the C call
   (e.g. `name.malloc(len)`, `pLogBuffer.malloc(bufferSize)`), so the binding
   owns the allocation;
3. the synthesis path emits `cdef CStr <buf> = CStr.fromPtr(NULL)`, the
   prepend `malloc`s it **under the GIL** before the `nogil` block, the call
   passes `<char *><buf>._ptr`, and the `CStr` is appended to the returned
   tuple — see §4.2 (rank-1 `char *` ⇒ `CStr`).

The intent override and the `malloc` prepend are driven off the *same* map,
so they cannot drift: a buffer marked `OUT_CALLEE_ALLOCATED` without a
matching `malloc` would pass a NULL pointer to C.

Contrast a `char **` OUT such as `hipDrvGetErrorName`, where the callee points
the slot at its **own internal storage** — it is classified
`OUT_CALLEE_ALLOCATED` structurally by `string_z` (§3 #5), gets `CStr` via
§4.2, and correctly receives **no** `malloc` (it is not in `_CSTR_OUT_BUFFERS`).
Only the degree-1, binding-allocates buffers are in the map.

### 4.4 Exceptions and escape hatches

- **Single `char` OUT** (rare): give the parm an explicit rank-0 override in
  the recipe. At rank 0 the OUT codepath takes the scalar branch and emits
  `cdef char name` / `&name`.
- **Binary `char *` buffer** (not a string): override the parm via the
  recipe `ptr_complicated_type_handler` to `NDBuffer` / `Pointer`.
- **`char **` array-of-strings IN** (argv): left as `Pointer` by default;
  override to `ListOfBytes` where the array semantics matter.

**Chain-order convention** for intent rules:

  1. **Per-library hardcoded overrides** — pinpoint
     `(funcname, parm_idx)` rules in the wrapped function body
     (e.g. `class hip` overrides for specific HIP entry points,
     amdsmi's verb / name-set heuristics). Highest precedence —
     run first.
  2. **`documented_param_intent`** — slot 0 of the
     `_*_INTENT_CHAIN` tuples. Author-stated direction trumps
     structural inference because a documented `@param[in] T**`
     declares an array-of-pointers input that
     `double_indirection_out` would otherwise misclassify as OUT.
  3. **Structural rules** in their existing order
     (`double_indirection_out`, `opaque_typedef_is_handle`,
     `array_with_length_param`, `status_return_out_pointer`,
     `string_z`, `conservative`, default).

**Mutual exclusion**: `pointer_as_value` and `pointer_as_reference` give
opposite verdicts for `T*` non-const. A recipe must choose at most one.

**Modifier signals deliberately ignored** (worth documenting so future
contributors don't try to use them):

- `T *const` (top-level pointer const) — promise about the *pointer
  variable*, not the *data*.
- `restrict` — aliasing hint, no direction signal.
- `volatile` — memory-model qualifier, no direction or shape signal.

### 4.5 Numeric rank-1 buffers map to `ListOf*` by element kind

Once a pointer parameter is classified as rank-1 (a sized buffer, not a
scalar slot), the Cython complicated-type handler
(`CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER` in
`interfacegen/cython/_defaults.py`) picks the `rocm.bindings.util.types`
wrapper from the **innermost canonical clang `TypeKind`** of the pointee:

| Innermost `TypeKind` | C element type    | Wrapper              |
|----------------------|-------------------|----------------------|
| `INT`                | `int`             | `ListOfInt`          |
| `LONG`               | `long` (`off_t`/`hoff_t`/`ssize_t`/`int64_t`) | `ListOfLong` |
| `UINT`               | `unsigned`        | `ListOfUnsigned`     |
| `ULONG`              | `unsigned long` (`size_t`) | `ListOfUnsignedLong` |
| `CHAR_S`             | `char`            | `CStr` (see §4)      |
| `VOID` (degree ≥ 2)  | `void*` slot      | `ListOfPointer`      |

Anything without a matching branch falls through to the generic
`Pointer`. The `LONG` → `ListOfLong` row exists so signed-`long` buffers
(notably hipFILE's `hoff_t*`/`ssize_t*` offset params) are list-
constructible sequences, consistent with `size_t*` → `ListOfUnsignedLong`,
instead of an opaque `Pointer`.

## 5. Module layout

```
interfacegen/
└── support/
    └── recipes/
        ├── control.py        # ParmIntent, DEFAULT_PTR_PARM_INTENT,
        │                     # DEFAULT_PTR_RANK, @fallback decorator
        ├── generic.py        # one class per convention (§3 catalog)
        └── rocm.py           # per-library classes (hip, hipblas, …)
                              # with @fallback chains wired in
```

- **`control.fallback(*fallbacks)`** — chains rule callables. The wrapped
  function runs first; if it returns `None`, each fallback is tried in
  order. First non-`None` wins. Apply order on a class method:

  ```python
  class hip:
      @staticmethod                          # outside
      @fallback(*_RUNTIME_INTENT_CHAIN)      # inside
      def ptr_parm_intent(parm):
          ...
          return None  # defer
  ```

- **`generic.<convention>.ptr_parm_intent` / `ptr_rank`** — each class
  exposes these as static methods. They take the parm/node, return a
  verdict or `None`.

- **`rocm._RUNTIME_INTENT_CHAIN` / `_NUMERICAL_INTENT_CHAIN` /
  `_INPLACE_NUMERICAL_INTENT_CHAIN`** (and the corresponding `_RANK`
  tuples) — module-level chain composites used by per-library classes so
  the recipe sites stay short.

## 6. How to add a new convention

1. Add a class to `interfacegen/support/recipes/generic.py`. Mirror the
   shape of the existing classes:

   ```python
   class my_convention:
       @staticmethod
       def ptr_parm_intent(parm):
           if <trigger>:
               return ParmIntent.<verdict>
           return None  # defer

       @staticmethod
       def ptr_rank(node):
           ...
           return None
   ```

2. Append it to the appropriate chain tuple in `rocm.py` (or define a new
   chain template if the convention applies to a new library family).
3. Add a row to the table in §3 of this doc with the source citation.
4. Add a smoke test in `python/interfacegen/test/test_generic_recipes.py`
   covering the trigger and at least one negative case.

## 7. How to bind a new ROCm library

1. Create a class in `interfacegen/support/recipes/rocm.py`:
   ```python
   class mylib:
       @staticmethod
       def node_filter(node): ...
       @staticmethod
       @fallback(*_NUMERICAL_INTENT_CHAIN)   # pick a template
       def ptr_parm_intent(parm):
           # hardcoded (funcname, parm_idx) overrides only
           return None
       @staticmethod
       @fallback(*_NUMERICAL_RANK_CHAIN)
       def ptr_rank(node):
           return None
   ```
2. Pick a chain template based on the API style:
   - `_RUNTIME_*` — runtime APIs, scalar OUTs, mostly handle creation
     (hip, hiprtc, hipfile)
   - `_NUMERICAL_*` — BLAS/LAPACK-style with `(T*, n)` paired params
     (hipblas, hipsolver, hipsparse, hiprand)
   - `_INPLACE_NUMERICAL_*` — adds `pointer_as_reference` for in-place
     compute and recv-buffer semantics (hipfft, rccl)
3. Run the codegen, eyeball the generated bindings for the highest-traffic
   functions, and add `(funcname, parm_idx)` carve-outs as you find
   misclassifications. The recipe pattern in `rcd.ptr_parm_intent` (the
   `ncclCommInitAll` exceptions) is the canonical example.

## 8. Unclassifiable-pointer fallback policy

When every callable in the chain (including `DEFAULT_PTR_PARM_INTENT`)
returns `None`, the cython backend applies a final, deliberately permissive
fallback:

- **Intent**: `ParmIntent.INOUT`. INOUT doesn't lie about read-only-ness
  (which would let a caller corrupt a `const` buffer) nor about init-state
  (a fully-initialized buffer is always a valid argument to an INOUT slot).
- **Generated parameter type**: `rocm.bindings.util.types.Pointer`
  (`packages/rocm-bindings-core/.../util/types.pxd`). The generic pointer
  wrapper accepts other `Pointer` instances, `ctypes` pointers, `int`
  addresses — surrendering type-safety in exchange for letting the user
  wire up the call themselves.

The cython `Parm` exposes `effective_ptr_intent` (chain verdict, `None` →
`INOUT`) and `is_ptr_intent_unclassified` (true iff the fallback applied).

### Two-axis intent: direction vs. allocation

`ParmIntent` carries two orthogonal axes so consumers never destructure
the enum by hand:

- **direction** (`ParmIntent.direction`) — the coarse `IN` / `OUT` /
  `INOUT` data-flow vocabulary. This is what direction-only backends
  reason in (Fortran `intent(in/out/inout)`); they compare
  `parm.intent.direction` and ignore the allocation axis entirely.
- **allocated_by_callee** (`ParmIntent.allocated_by_callee`) — a boolean
  meaningful only for `OUT`. The Python/Cython layer reads it to decide
  caller-vs-callee allocation.

`OUT_CALLEE_ALLOCATED` is a strict refinement of `OUT` (it coarsens back
to `OUT` via `.direction`), distinguishing a fresh callee-produced
scalar / opaque handle / NUL-terminated string from a caller-allocated
buffer the callee merely fills. The doxygen `@param[out]` tag cannot
express this difference, which is exactly why a dedicated enum value is
needed.

**`OUT_CALLEE_ALLOCATED` is a best-effort hint, layered on top of the
coarse `IN`/`OUT`/`INOUT` direction.** Rules should provide it wherever
they can prove callee-allocation — the structural producers
(`double_indirection_out` for `T**`, `string_z` for `char**`) and the
documented rank-0/double-indirection promotion in
`documented_param_intent`. A binding generator is free to treat it as
plain `OUT` by reading `.direction`; direction-only backends (Fortran)
do exactly that and ignore the allocation axis entirely.

**The allocation axis is stated by the rules, not re-derived from rank.**
`@param[out]` denotes output *direction* only — it says nothing about who
allocates. A caller-allocated output buffer (e.g. `hipMemcpy`'s `dst`, a
rank-1 `void*`) is correctly tagged `[out]` and stays in the args. The
recipe layer prescribes the two real, language-agnostic properties —
direction (`IN`/`OUT`/`INOUT`) and rank (0 = single slot, ≥1 = sized
buffer) — plus the explicit `OUT_CALLEE_ALLOCATED` hint wherever a rule
can prove the callee produces the value. The Cython consumer then reads
*only* that hint:

```python
is_out_callee_allocated_ptr = intent.allocated_by_callee
```

There is deliberately **no rank-0 fallback**. Caller-allocated `IN`,
`INOUT`, and `OUT` scalars are all handled the same way — they stay
pointer arguments (a rank-0 `PointerTo*`, a rank-1 `ListOf*`) — and only
an explicit `OUT_CALLEE_ALLOCATED` becomes a synthesized return. This is
what lets a caller-*provided* rank-0 `OUT` scalar the callee writes later
stay an argument: the canonical case is hipFILE's async `bytes_read_p` /
`bytes_written_p` (`ssize_t*`, documented `@param[out]`), whose stream
writes them *after* the call returns, so the caller must keep the pointer
and read it post-synchronization. The `hipfile` recipe pins these to plain
`OUT` (ahead of `documented_param_intent` in the chain) and leaves them at
their honest rank 0, so they render as `PointerToLong` arguments.

Consequently, a rule that wants a rank-0 scalar `OUT` to become a return
must *say so* by returning `ParmIntent.OUT_CALLEE_ALLOCATED`. The
structural producers already do (`double_indirection_out` for `T**`,
`string_z` for `char**`, `documented_param_intent` for a documented
`[out]` on a callee-allocated shape); a per-library hardcode that pins a
bare rank-0 `T*` / `record*` / `enum*` scalar `OUT` states the hint
directly (there is no structural signal for a single `T*`'s direction, so
it cannot be inferred by a chain rule).

Because the fallback is gone, the earlier `_HIPMEMCPY_RECORD_DST_NAMES`
rank-override-to-1 is no longer load-bearing for *allocation* (a rank-0
`record*` destination pinned to plain `OUT` already stays a caller-
allocated argument); it is retained only where it still matters for the
wrapper/shape choice. The `void*`-alias destinations (`hipDeviceptr_t`)
remain rank-1 buffers via `opaque_typedef_is_handle`.

**Callee-allocation is decoupled from rank and from the wrapper type.**
A callee-allocated parameter can be a scalar, a handle, *or* a buffer:
`hipMalloc(void** ptr, size)` is callee-allocated and rank-1, rendered
by the complicated-type handler as a `DeviceArray` (a byte sequence),
not a scalar. Who allocates (the allocation axis) and what Python type
is returned (the wrapper) are separate decisions — which is why the
hint must be stated by a rule and preserved for such sites.

Both backends consume the effective verdict through `parm.intent` and
branch on `parm.intent.direction` directly. The cython `Parm` keeps two
convenience predicates over that verdict — `is_out_ptr`
(`intent.direction == OUT`) and `is_out_callee_allocated_ptr`
(`intent.allocated_by_callee`) — because the cython OUT dispatch reads
them; the IN / INOUT cases need no predicate (they are the dispatch's
default branch).

### Producers of `OUT_CALLEE_ALLOCATED`

The hint is no longer the output of a single rule — it is emitted from three
layers, consulted in the precedence order of §4.4 (per-library hardcodes
first, then `documented_param_intent`, then the structural rules). Any
producer that can *prove* callee-allocation should emit it; a consumer that
doesn't care reads `.direction` and sees plain `OUT`.

**1. Generic structural rules** (`support/recipes/generic.py`, library-agnostic):

| Rule (§3) | Trigger | Rank emitted |
|-----------|---------|-------------:|
| `double_indirection_out` (#4) | non-const `T**` — `void**`, `record**`, `enum**`, `basic**` | 0 for a typed handle slot; **1 for `void**`** (the `hipMalloc` byte-buffer idiom) |
| `string_z` (#5) | non-const `char **` (returned string) | 1 |
| `documented_param_intent` (#10) | a doxygen `@param[out]` **on a callee-allocated shape** — the `_is_callee_allocated_out_shape` test: rank-0 scalar/handle/string, OR a non-const `T**`/`void**`, OR a `char**`. A `[out]` on a caller-sized `T*` buffer stays plain `OUT`. | inherited |

**2. Per-library hardcoded overrides** (`support/recipes/rocm.py`, highest
precedence — these run before the structural chain):

- **HIP opaque-handle creators** — `_HIP_HANDLE_CREATOR_OUT_PARM0` (parm 0)
  and `_HIP_HANDLE_CREATOR_OUT_PARM01` (parms 0 and 1): `hipStreamCreate`,
  `hipEventCreate`, `hipMalloc*`, `hipModule{Load,Get}*`, `hipMemPool*`,
  `hipGraph*`, `hipCtxCreate`, … Doxygen mistags these `@param[in, out]`;
  the override restores them to callee-allocated OUT (see UPSTREAM_BUGS
  Family 2).
- **HIP `void**` named slots** — a `void**` named `devPtr` / `ptr` /
  `dev_ptr` / `data` / `dptr` is a handle-creation slot.
- **HIP scalar-via-pointer OUTs** — `pointer-to-enum` (degree 1), and
  non-`char` `pointer-to-basic-type` (degree 1): the callee writes a fresh
  scalar through the slot. (These will be subsumed by
  `status_return_out_pointer` (#8) once that relational rule lands.)
- **hipBLAS / hipSOLVER handle creators** — a `void**` named `handle`
  (`hipblasCreate`, `hipsolverCreate`).
- **Per-library rank-0 scalar/handle OUTs** — bare `T*` / `record*` /
  `enum*` scalar OUTs that a rule knows are callee-produced now state the
  hint directly (there is no structural signal for a single `T*`'s
  direction). Examples: HIP `hipDeviceGetUuid` / `hipIpcGetMemHandle`;
  hipRTC `hiprtcVersion` (`major`/`minor`), `*SizeRet`, `hip_link_state_ptr`,
  `size_out`; RCCL `ncclGetUniqueId` and basic-scalar OUTs; hipFFT
  `workSize`; hipSPARSE `hipsparseCreate`; amdsmi `_MISTAGGED_OUT` and the
  shape-aware `amdsmi_get_*` verb catch-all (via
  `generic.is_callee_allocated_out_shape`).
- **`T**` handle creators defer to the chain** — the redundant per-library
  `is_pointer_to_record(degree=2)` hardcodes (RCCL `comm`, hipRAND, hipFFT
  `plan`, hipSPARSE descriptor creators) were removed; `double_indirection_out`
  in the shared chain already classifies non-const `T**` as
  `OUT_CALLEE_ALLOCATED`.

**3. Cython-only recipe-generator overrides** (`generators_hip.py`):

- the binding-allocated `char *` string buffers in `_CSTR_OUT_BUFFERS`
  (`hipDeviceGetName.name`, `hipDeviceGetPCIBusId.pciBusId`,
  `hipGraphInstantiate.pLogBuffer`; §4.3). These are deliberately **absent
  from `controls.hip`** so that direction-only backends keep them as plain
  caller-allocated `OUT`; only the HIP Cython backend promotes them.

The first two layers feed the shared intent chain consumed by every backend;
the third wraps that chain for the HIP Cython generator alone. All three
ultimately set the same `ParmIntent.OUT_CALLEE_ALLOCATED` value, which the
Cython dispatch below turns into a synthesized return.

### Cython dispatch: two pointer paths keyed on allocation

`Function._analyze_parms` routes each pointer parm down one of two
handlers, keyed purely on `is_out_callee_allocated_ptr` (which already
implies direction `OUT`):

- **`handle_callee_allocated_ptr_parm`** — the *only* path that adds a
  return-tuple entry. Reached solely by `OUT_CALLEE_ALLOCATED`: the
  callee produces a fresh handle / scalar / string and the codegen
  synthesizes it as a return value. If the shape can't be synthesized it
  raises `CodegenUnsupportedPattern`; the dispatcher rolls the entry back
  and degrades to the caller-allocated path (the param stays a plain OUT
  pointer, the return tuple is unchanged, and the caller passes a
  `Pointer`).
- **`handle_caller_allocated_ptr_`** — the shared path for **IN**, plain
  caller-allocated **OUT**, and **INOUT**. The caller owns the buffer, so
  it never adds a return-tuple entry. It is *total*: any shape the
  structured branches don't bind falls back to the generic `Pointer`
  wrapper (via `ptr_complicated_type_handler`) instead of crashing.

So a plain (caller-allocated) `OUT` buffer is bound as a caller-passed
argument, while a callee-allocated `OUT` becomes a return value — the
concrete payoff of the orthogonal allocation axis.

### Always-on: original C signature in every generated docstring

Independent of the fallback path, every generated function binding's
docstring contains a `C signature` section showing the verbatim original
declaration:

```text
C signature
-----------
    hipError_t hipStreamCreate(hipStream_t* stream)
```

The user needs this whether or not a parm fell through — to
cross-reference ROCm documentation, to understand what they're passing
when an `INOUT` `Pointer` shows up, and to verify any ctypes wiring
matches the C ABI.

## 9. Prior art: annotation systems and conventions

C deliberately erases the intent/rank information interfacegen needs (§1), so
none of it can be recovered from the language alone. Other ecosystems hit the
same wall and bolted a *parameter-annotation layer* on top of C to recover it
— for static analysis, for RPC marshalling, or for auto-generating
cross-language bindings (exactly our problem). The conventions cataloged in §3
are distilled from these systems; the "Sourced from" column there cites them by
the shorthands defined below. Where a system and interfacegen disagree on a
term, this section is the glossary of record.

### 9.1 SAL — Microsoft Source-code Annotation Language

SAL is a set of macros (`_In_`, `_Out_`, `_Inout_`, `_Outptr_`,
`_In_reads_(n)`, `_Out_writes_(n)`, `_In_z_`, `_Outptr_result_z_`, …) that
annotate C/C++ function parameters with direction, buffer extent, and
nullability. They expand to nothing in a normal build and are consumed by the
MSVC `/analyze` static analyzer. SAL is the closest match to interfacegen's
model because it separates the same axes we do: `_In_`/`_Out_`/`_Inout_` is
our **direction**, `_..._reads_(n)`/`_..._writes_(n)` is our **rank** (a sized
buffer), `_Outptr_` is our `OUT_CALLEE_ALLOCATED` double-indirection case, and
the `_z_` family is our `string_z` NUL-terminated-string convention.

- *Understanding SAL*: <https://learn.microsoft.com/en-us/cpp/code-quality/understanding-sal>
- *Annotating function parameters and return values*: <https://learn.microsoft.com/en-us/cpp/code-quality/annotating-function-parameters-and-return-values>

### 9.2 GIR — GObject Introspection annotations

GObject Introspection (GIR) is the GLib/GTK machinery that scrapes annotated C
headers to auto-generate bindings for Python, JavaScript, Rust, and others —
the same end goal as interfacegen. Its in-comment annotations
(`(in)`, `(out)`, `(inout)`, `(array length=N)`, `(array zero-terminated=1)`,
`(transfer …)`, `(nullable)`) map directly onto our conventions: `(out)` on a
double-indirection parameter is our `double_indirection_out`,
`(array length=N)` is `array_with_length_param`, and
`(array zero-terminated=1)` is `zero_terminated_array`. The `(transfer …)`
ownership annotation has no direct interfacegen analogue but informs who frees
callee-allocated returns.

- GObject Introspection annotation reference: <https://gi.readthedocs.io/en/latest/annotations/giannotations.html>

### 9.3 COM / IDL (MIDL) — directional parameter attributes

COM interfaces are described in Microsoft Interface Definition Language (MIDL),
where every parameter is tagged `[in]`, `[out]`, or `[in, out]` (plus
extent attributes `[size_is]`, `[length_is]`, `[max_is]`). The MIDL compiler
uses these to generate RPC marshalling stubs — it must know direction to know
which way to copy bytes across a process boundary, the same reason a binding
generator must. A MIDL-relevant rule we lean on: **all `[out]` parameters must
be pointers** (you can't return through a by-value argument in C), and a
callee-allocated `[out]` handle must be a pointer-to-pointer — precisely the
shape interfacegen classifies as `OUT_CALLEE_ALLOCATED`.

- `[in]` attribute: <https://learn.microsoft.com/en-us/windows/win32/midl/in>
- `[out]` attribute: <https://learn.microsoft.com/en-us/windows/win32/midl/out-idl>
- Anatomy of an IDL file (worked `[in]`/`[out]`/`[in, out]` example): <https://learn.microsoft.com/en-us/windows/win32/com/anatomy-of-an-idl-file>

### 9.4 C++ Core Guidelines and the "reference vs. value" schools

These are *style* guidance rather than machine-readable annotation systems, but
they are the source of conventions #2 and #3 (`pointer_as_reference` vs.
`pointer_as_value`) — the two opposed readings of a bare non-const `T*`. The
C++ Core Guidelines (F.15–F.21) codify the "outputs leave via the return value,
in-out via non-const reference" position; Ropert's article surveys the
reference/pointer/value trade-off and is why the doc names the two camps.

- C++ Core Guidelines, F.15–F.21 (parameter-passing conventions): <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f-call>
- Mathieu Ropert, *Input-output arguments: reference, pointers or values?*: <https://mropert.github.io/2018/04/03/output_arguments/>

### 9.5 The status-return / OUT-pointer idiom

Not a formal annotation system but a near-universal C convention underpinning
`status_return_out_pointer` (#8): the function returns an integer/enum status
code and delivers its real result through an OUT pointer. POSIX, every Khronos
API (OpenGL, Vulkan, OpenCL), and every ROCm runtime entry point
(`hipError_t hipFoo(…, T* out)`) follow it. interfacegen exploits the inverse:
a single non-const pointer parameter on a status-returning function is almost
always that OUT slot.
