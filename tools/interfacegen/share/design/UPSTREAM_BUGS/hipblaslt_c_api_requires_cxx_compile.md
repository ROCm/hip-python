# Bug report: `hipblaslt.h` claims to be a C API but cannot be compiled as C

**Filed against:** [`ROCm/rocBLAS-internal` →
hipBLASLt](https://github.com/ROCm/rocBLAS-internal) (hipBLASLt
public C-API headers)

**Encountered on:** 2026-05-12

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version`
reports `7.13.0`; line numbers below are from this checkout).
Same shape on every release that ships `hipblaslt.h` with
unconditional `<memory>` / `<regex>` / `<vector>` includes.

**Affected files:**
- `/opt/rocm/include/hipblaslt/hipblaslt.h` (umbrella header)
- `/opt/rocm/include/hipblaslt/hipblaslt-types.h` (transitively
  included)

## Title

`hipblaslt.h` declares an `extern "C" { … }` API surface but
unconditionally `#include`s C++ standard-library headers and the
C++-only `hip_bfloat16` struct, so any C translation unit that
`#include`s it fails to compile.

## Summary

A correctly-shaped C API header is intended to be includable from
both C and C++ translation units — that is the point of the
`#ifdef __cplusplus extern "C" { #endif` wrapper that hipBLASLt
uses. `hipblaslt.h` violates this in two independent ways:

### 1. Three C++ stdlib headers included unconditionally and unused

```c
// /opt/rocm/include/hipblaslt/hipblaslt.h:54-56
#include <memory>
#include <regex>
#include <vector>
```

These three lines are at file scope, not gated by
`#ifdef __cplusplus`. `<memory>`, `<regex>`, `<vector>` are C++
standard-library headers — there is no C equivalent on the
preprocessor's search path. A plain-C compile fails immediately:

```
hipblaslt.h:54:10: fatal error: memory: No such file or directory
   54 | #include <memory>
      |          ^~~~~~~~
compilation terminated.
```

A `grep` over the entire `hipblaslt/` install tree confirms
neither `std::unique_ptr` / `std::shared_ptr`, nor `std::regex`
/ `std::regex_match`, nor `std::vector` is referenced anywhere in
the public headers. The three includes appear to be vestigial:

```sh
$ grep -rE 'std::(unique_ptr|shared_ptr|make_unique|make_shared|regex|vector)' \
       /opt/rocm/include/hipblaslt/
$ # (no output)
```

### 2. `typedef hip_bfloat16 hipblasLtBfloat16` with no C-only fallback

```c
// /opt/rocm/include/hipblaslt/hipblaslt-types.h:69
typedef hip_bfloat16 hipblasLtBfloat16;
```

The aliased `hip_bfloat16` is declared in
`/opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h` as a C++
struct with constructors, conversion operators, and an
`enum truncate_t { truncate };` member:

```cpp
// /opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h:38-46
struct hip_bfloat16 {
  __hip_uint16_t data;

  enum truncate_t { truncate };

  __HOST_DEVICE__ hip_bfloat16() = default;

  // round upper 16 bits of IEEE float to convert to bfloat16
  explicit __HOST_DEVICE__ hip_bfloat16(float f) : data(float_to_bfloat16(f)) {}
  …
};
```

C does not allow `enum`-inside-struct, default member functions,
or member initializer lists, so `<hip/hip_bfloat16.h>` cannot be
parsed by a C compiler.

For comparison, the **sibling library `hipblas` does this
correctly**: `hipblas/hipblas.h:5614-5634` gates the C++-only
typedef path behind `#if defined(HIPBLAS_USE_HIP_BFLOAT16)` and
falls back to a C-compatible POD struct otherwise:

```c
// /opt/rocm/include/hipblas/hipblas.h
#if defined(HIPBLAS_USE_HIP_BFLOAT16)
#include <hip/hip_bfloat16.h>
typedef hip_bfloat16 hipblasBfloat16;
#elif __cplusplus < 201103L || !defined(HIPBLAS_BFLOAT16_CLASS)
// minimal POD definition that compiles in plain C
typedef struct hipblasBfloat16 {
    uint16_t data;
} hipblasBfloat16;
#else
class hipblasBfloat16 { … };
#endif
```

`hipblaslt-types.h:69` has no equivalent fallback — the typedef
is unconditional.

## Reproducer

```sh
$ cat > /tmp/test_c.c <<EOF
#include <hipblaslt/hipblaslt.h>
int main(void) { return 0; }
EOF
$ gcc -I/opt/rocm/include -c /tmp/test_c.c -o /tmp/test_c.o
/opt/rocm/include/hipblaslt/hipblaslt.h:54:10: fatal error:
    memory: No such file or directory
   54 | #include <memory>
      |          ^~~~~~~~
compilation terminated.
```

If the three stdlib includes are commented out, the next failure
is `hip_bfloat16`'s C++-only constructors:

```
/opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h:43:42:
    error: expected ';', identifier, or '(' before '=' token
   43 |   __HOST_DEVICE__ hip_bfloat16() = default;
      |                                  ^
```

## Impact

Any project that needs to consume hipBLASLt as a C API is
blocked. Concrete examples:

- **Binding generators** that parse C headers with libclang in
  C-mode (`-x c`) or compile C wrappers — for example, every
  Python / Rust / Go / OCaml / Lua / Lisp wrapper for hipBLASLt
  has to either patch the header, vendor a private "C-only"
  copy, or compile its glue as C++. None of these are necessary
  for hipBLAS, hipFFT, hipSPARSE, hiprand, hipsolver, hipdnn —
  all of which are usable from a plain C translation unit today.
- **Embedding via FFI in C-only host languages** (e.g. classic C,
  embedded toolchains where the C++ runtime is not linked).
- **Lightweight `cargo build` / `cmake` consumers** that don't
  pull in libstdc++ for unrelated reasons.

The `extern "C" { … }` wrapper around the function declarations
proves the API surface is *intended* to be C-callable. The two
issues above prevent that intent from being realized.

## Suggested fix

1. **Remove the three unused includes** from `hipblaslt.h:54-56`:

   ```diff
   - #include <memory>
   - #include <regex>
   - #include <vector>
   ```

   They are not referenced anywhere in the public headers.

2. **Add a C-compatible fallback** for `hipblasLtBfloat16` in
   `hipblaslt-types.h:69`, mirroring the hipBLAS pattern:

   ```c
   #if defined(HIPBLASLT_USE_HIP_BFLOAT16)
       #include <hip/hip_bfloat16.h>
       typedef hip_bfloat16 hipblasLtBfloat16;
   #elif __cplusplus < 201103L || !defined(HIPBLASLT_BFLOAT16_CLASS)
       /* minimal POD definition for plain-C consumers */
       typedef struct hipblasLtBfloat16 {
           uint16_t data;
       } hipblasLtBfloat16;
   #else
       class hipblasLtBfloat16 { … };
   #endif
   ```

Either fix in isolation would be a net improvement; both together
make `hipblaslt.h` properly C-includable.

## Workaround for downstream

Patch the header on the downstream side — comment out the three
unused stdlib includes, replace the `hipblasLtBfloat16` typedef
with a POD struct, and `#define HIPBLASLT_BFLOAT16_CLASS` (or a
similar guard the patched header recognises) to keep C++
consumers building.

This is invasive enough that no production downstream is doing
it; instead, the standard workaround is to bind hipBLASLt at the
C++ source level, paying the cost of pulling in libstdc++.

## See also

The companion bug report
[`hipsparselt_c_api_requires_cxx_compile.md`](hipsparselt_c_api_requires_cxx_compile.md)
covers the same shape of problem for `hipsparselt.h` (the
unconditional `#include <hip/hip_bfloat16.h>` is shared; the
unused-stdlib-include subset is hipBLASLt-specific).
