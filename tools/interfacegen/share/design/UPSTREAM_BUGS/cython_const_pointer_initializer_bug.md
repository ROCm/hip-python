# Bug report: Cython 3.0.x silently miscompiles `cdef T x = <T>expr` for `*const *` types

**Filed against:** [`cython/cython`](https://github.com/cython/cython)

**Encountered on:** 2026-05-08

**Affected Cython versions:** **3.0.x** (verified on 3.0.12, the latest
3.0 release at the time of this report). Fixed upstream in **3.1.0**;
not present in 3.1.x or 3.2.x.

**Reproducer:** `share/design/UPSTREAM_BUGS/test_const_bug.pyx`
together with `run_test.sh` (static C-source check) and
`runtime_verify.sh` (compiles to `.so` and reads back the assigned
value). Both scripts iterate over every Cython version installed at
`/tmp/cython_versions/cy<X.Y.Z>/` and contrast the buggy and the
working shapes.

## Title

`cdef T x = <T>expr` silently drops the initializer when `T` contains
the inner `*const *` pattern (e.g. `const char *const *`,
`void *const *`).

## Summary

Cython 3.0.x parses the `cdef`-with-initializer statement and emits
the warning

    local variable 'x' referenced before assignment

— then proceeds to **emit only the C declaration** of the local,
**dropping the initializer assignment**. The generated C code has

```c
char const *const *__pyx_v_x;   // declared, never assigned
```

with no `__pyx_v_x = ...` statement anywhere in the function body.
At runtime the local takes whatever value the stack happened to hold
at function entry — typically `NULL` because the calling convention
clears the relevant registers on entry, but in principle indeterminate.

The same shape applies to `cdef void *const x = <void *const>expr`
(trailing-const single pointer): 3.0.x silently drops the initializer
and warns; 3.1+ has changed behaviour and now hard-errors with
`Assignment to const 'x'` for that case (which is also a problem,
but at least diagnoses the issue instead of producing a NULL-deref
binary).

The pointer-to-const-pointer (`*const *`) case is the dangerous one
because (a) 3.0.x compiles it without error, (b) the warning is one
line of noise easily lost amid a typical Cython build's output, and
(c) the generated `.so` works fine for any code path that doesn't
read the local — the crash only surfaces when something downstream
dereferences the (NULL) value, which can be much later than the
build step that produced the broken `.so`.

## Reproducer

A self-contained `test_const_bug.pyx` declares six variants of the
cdef-with-initializer pattern; the runner scripts (`run_test.sh` for
static C-source assertion, `runtime_verify.sh` for compile-and-call)
report which variants get an assignment in the generated C and which
return a non-NULL pointer at runtime.

```python
from libc.stdlib cimport malloc, free


cdef class W:
    cdef void* _ptr
    def __cinit__(self):
        self._ptr = malloc(8)
    def __dealloc__(self):
        if self._ptr:
            free(self._ptr)
    cdef void* getPtr(self):
        return self._ptr


# Bug shape: dropped initializer.
def buggy_const_T_const_pp(W w):
    cdef const char *const * x = <const char *const *>w.getPtr()
    return <long>x

def buggy_T_const_pp(W w):
    cdef void *const * x = <void *const *>w.getPtr()
    return <long>x

# Working shape: split form (bare cdef + separate assignment).
def workaround_split_const_T_const_pp(W w):
    cdef const char *const * x
    x = <const char *const *>w.getPtr()
    return <long>x

# Working shape: drop the inner const from the cdef type.
def ok_const_T_pp(W w):
    cdef const char ** x = <const char **>w.getPtr()
    return <long>x

# Working shape: no const at all.
def ok_T_pp(W w):
    cdef void ** x = <void **>w.getPtr()
    return <long>x

# Working shape: single-level const pointee.
def ok_const_T_p(W w):
    cdef const char * x = <const char *>w.getPtr()
    return <long>x
```

Cross-version results
(`runtime_verify.sh` returns the wrapper's heap pointer — non-zero
means the initializer worked):

| Cython | C-source assignment for buggy shapes | Runtime returned pointer |
|--------|--------------------------------------|--------------------------|
| 3.0.12 | **missing**; warning ``referenced before assignment`` emitted | **NULL** (silent miscompile) |
| 3.1.0  | present; no warnings | correct non-NULL |
| 3.1.8  | present; no warnings | correct non-NULL |
| 3.2.4  | present; no warnings | correct non-NULL |

The non-buggy shapes (`const T **`, `T **`, `const T *`) are
correctly compiled across all four versions.

## Why this matters

The C signatures `const char *const *` and `T *const *` are common
in C library APIs whenever an argument is a *pointer to an
immutable array of immutable strings* (or, more generally, an
array-of-pointers parameter where neither the array slots nor the
pointed-to data should be mutated by the callee). The natural
Cython prehoist for such an argument

```cython
cdef <wrapper_class> obj = <wrapper_class>.fromPyobj(py_arg)
cdef const char *const * raw = <const char *const *>obj.getPtr()
with nogil:
    c_func(raw)   # <-- raw is NULL here on Cython 3.0.x
```

results in `raw == NULL` at the call site under Cython 3.0.x, and
any C library that walks the array on the first iteration will
SIGSEGV inside its own code — making the bug look like a C-library
defect rather than a Cython one.

## Workarounds

Either of the following compiles correctly on Cython 3.0.x:

1. **Split form: bare cdef + separate assignment.** The bug is
   specifically in the cdef-with-initializer syntactic position;
   a stand-alone assignment statement is unaffected.

   ```cython
   cdef const char *const * raw
   raw = <const char *const *>obj.getPtr()
   ```

2. **Drop the inner const from the cdef type.** The C function's
   parameter type still requires `const char *const *`, but a
   Cython local doesn't need that protection — the wider type
   `const char **` (or even `void **`) accepts the cast result and
   the resulting pointer is still passable to the C function (any
   discarded const is restored by the implicit conversion at the
   call site).

   ```cython
   cdef const char ** raw = <const char **>obj.getPtr()
   ```

For the trailing-const case (`cdef void *const x = ...`), the
trailing const must be stripped from the cdef local type — the
split form fails on 3.1+ with `Assignment to const 'x'` and 3.0.x
silently drops the initializer either way. The cast on the rhs can
keep the original type, the assignment is still type-correct
against any C function signature that requires the const.

## Former code-generator workaround (removed; superseded by the Cython >= 3.1.0 floor)

For a period the code generator implemented workaround #1 (split
form) automatically: the with-nogil call-arg hoist renderer
(`CallArgHoist.render_prehoist` in
`python/interfacegen/cython/_defaults.py`) detected the `*const *`
shape and emitted a bare `cdef` declaration plus a separate
assignment instead of the combined cdef-with-initializer.

That branch was **removed** once the project committed to a
Cython >= 3.1.0 build floor (pinned in every requirements/pyproject
file): 3.1+ compiles the combined form correctly (see the
cross-version table above), so the split is no longer needed and the
combined form is more readable. The trailing-const strip (the second
transformation below) is **retained** — Cython 3.1+ still hard-errors
on a const-qualified local (`Assignment to const 'x'`).

The removed logic, preserved here so it can be revived if the floor
is ever lowered:

```python
import re

# Matches an *inner* const on a dereferenced pointer — the dangerous
# `*const *` shape (asterisk, const, whitespace, asterisk). A trailing
# const (e.g. `void *const` with no following `*`) does NOT match.
_NEEDS_SPLIT_FORM = re.compile(r"\*\s*const\s+\*")

# Strip a trailing `const` from the END of a c_type (e.g. `void *const`
# -> `void *`). RETAINED in the current generator.
_STRIP_TRAILING_CONST = re.compile(r"\s*\bconst\b\s*$")

def render_prehoist(self, arg_name: str) -> str:
    if self.plain_expr is not None:
        return f"cdef {self.c_type} {arg_name} = {self.plain_expr}"
    obj_name = f"{arg_name}_obj"
    prefix = f"cdef {self.wrapper_class} {obj_name} = {self.wrapper_factory}\n"
    rhs = f"{self.cast_open}{obj_name}.{self.pointer_extract}{self.cast_close}"
    cdef_type = self._STRIP_TRAILING_CONST.sub("", self.c_type)
    if self._NEEDS_SPLIT_FORM.search(cdef_type):
        # Bug-shape: emit bare cdef + separate assignment so
        # Cython 3.0.x emits the assignment in the C output.
        return (
            f"{prefix}"
            f"cdef {cdef_type} {arg_name}\n"
            f"{arg_name} = {rhs}"
        )
    return f"{prefix}cdef {cdef_type} {arg_name} = {rhs}"
```

Example contrast for the `const char *const *` options argument of
`hiprtcCompileProgram` (wrapper-bound hoist):

```cython
# Former split form (emitted for the `*const *` shape on 3.0.x):
cdef rocm.bindings.util.types.ListOfBytes _cy_f__arg_2_obj = rocm.bindings.util.types.ListOfBytes.fromPyobj(options)
cdef const char *const * _cy_f__arg_2
_cy_f__arg_2 = <const char *const *>_cy_f__arg_2_obj.getPtr()

# Current combined form (safe on Cython >= 3.1.0):
cdef rocm.bindings.util.types.ListOfBytes _cy_f__arg_2_obj = rocm.bindings.util.types.ListOfBytes.fromPyobj(options)
cdef const char *const * _cy_f__arg_2 = <const char *const *>_cy_f__arg_2_obj.getPtr()
```

## Suggested upstream fix

The bug appears to be in Cython's analysis of `cdef`-with-initializer
when the declared type is a multi-level pointer with an inner `const`
qualifier — the type parser should accept the form and the codegen
should emit the assignment, the same way it does for `const T **`
(no inner const) or `T **` (no const at all).

For the trailing-const case (`T *const`), the appropriate fix is
either to (a) accept the cdef-with-initializer consistently (Cython
already accepts `cdef const int x = i` in some contexts and rejects
it in others — the check is not uniform across the type lattice),
or (b) emit a clearer error message earlier, calling out the
specific incompatible form rather than the generic
`Assignment to const 'x'` which fires on a downstream node.

The reproducer in this directory is small (single `.pyx` file, two
runner scripts) and self-contained — should be straightforward to
add as a Cython regression test once the underlying analyser bug is
identified.
