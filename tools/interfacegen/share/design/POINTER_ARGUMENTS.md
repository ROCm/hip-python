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
| 4 | **double_indirection_out**          | non-const `T**` (esp. `void**`, `struct**`, `enum**`)                                           | OUT, rank 0 (callee-allocates)           | GIR `(out)` for double-indirection on a structure parameter; SAL `_Outptr_`. COM `[out]` requires a pointer. |
| 5 | **string_z**                        | `const char *` ⇒ IN, scalar; `char *` ⇒ ambiguous; `char **` ⇒ OUT scalar                       | as listed                                | GIR / SAL `_In_z_`, `_Outptr_result_z_` |
| 6 | **array_with_length_param**         | `T *buf` adjacent to integer parm whose name names the length (`n`, `len`, `count`, `*_size`)  | `buf` is rank 1; intent follows pointee const | GIR `(array length=N)`. SAL `_In_reads_(n)` / `_Out_writes_(n)`. *(deferred — relational rule)* |
| 7 | **zero_terminated_array**           | `T**` where elements are sentinel-terminated                                                    | rank 1                                   | GIR `(array zero-terminated=1)`. *(deferred)* |
| 8 | **status_return_out_pointer**       | function return type is an integer/error status enum AND parm is the only non-const pointer parm | that parm is OUT                         | C tradition: status code as return value, results via OUT pointer (POSIX, every Khronos API, every ROCm runtime). *(deferred — needs return-type access)* |
| 9 | **opaque_typedef_is_handle**        | type is a typedef whose canonical form is `T*`                                                   | rank 0 (the typedef is a scalar handle)  | Universal opaque-handle pattern (`FILE*`, `cl_context`, `VkInstance`, `cudaStream_t`, `hipStream_t`) |
| 10 | **documented_param_intent**         | parent function's raw doxygen comment carries `@param[in\|out\|in,out] <pname>` (or `\param[…]`) | the documented intent (IN / OUT / INOUT) | Doxygen direction tags — explicit author intent. Originated as `amdsmi._doxygen_intent`; an audit found it resolved 88 mismatches a verb-based heuristic produced for amdsmi.h alone. |

Conventions #6–8 are sketched in `generic.py` with TODO bodies; they need
relational access (sibling parms, function return type) that the
`tree.Parm` API doesn't yet expose ergonomically. They're slated for a
follow-up.

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

## 4. Module layout

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

## 5. How to add a new convention

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

## 6. How to bind a new ROCm library

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

## 7. Unclassifiable-pointer fallback policy

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

## 8. Pointers / further reading

- Microsoft SAL — *Understanding SAL*: <https://learn.microsoft.com/en-us/cpp/code-quality/understanding-sal>
- GObject Introspection annotations: <https://gi.readthedocs.io/en/latest/annotations/giannotations.html>
- C++ Core Guidelines, F.15–F.21 (in/out conventions): <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>
- Mathieu Ropert, "Input-output arguments: reference, pointers or values?"
- COM/IDL `[in]`, `[out]` directional attributes
