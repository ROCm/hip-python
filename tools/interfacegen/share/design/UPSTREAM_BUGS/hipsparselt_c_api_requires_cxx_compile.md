# Bug report: `hipsparselt.h` claims to be a C API but cannot be compiled as C

**Filed against:** [`ROCm/hipSPARSELt`](https://github.com/ROCm/hipSPARSELt)
(hipSPARSELt public C-API headers)

**Encountered on:** 2026-05-12

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version`
reports `7.13.0`; line numbers below are from this checkout).
Same shape on every release that ships `hipsparselt.h` with an
unconditional `#include <hip/hip_bfloat16.h>` under
`__HIP_PLATFORM_AMD__`.

**Affected files:**
- `/opt/rocm/include/hipsparselt/hipsparselt.h` (umbrella header)
- `/opt/rocm/include/hip/hip_bfloat16.h` →
  `/opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h`
  (transitively included; the actual C++-only source)

## Title

`hipsparselt.h` declares an `extern "C" { … }` API surface but
unconditionally `#include`s the C++-only `hip_bfloat16` struct
(via `hip/hip_bfloat16.h`), so any C translation unit that
`#include`s it fails to compile.

## Summary

`hipsparselt.h` wraps its function declarations in `extern "C"`
and exposes only POD parameter types — exactly the shape of a
C-callable API. But the umbrella header includes `hip_bfloat16.h`
unconditionally on AMD platforms:

```c
// /opt/rocm/include/hipsparselt/hipsparselt.h
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
…
#elif defined(__HIP_PLATFORM_NVIDIA__)
typedef __nv_bfloat16 hip_bfloat16;
…
#endif
```

The included `hip/hip_bfloat16.h` resolves to
`hip/amd_detail/amd_hip_bfloat16.h`, which declares
`hip_bfloat16` as a C++ struct with constructors and member
operators:

```cpp
// /opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h:38-46
struct hip_bfloat16 {
  __hip_uint16_t data;

  enum truncate_t { truncate };

  __HOST_DEVICE__ hip_bfloat16() = default;

  // round upper 16 bits of IEEE float to convert to bfloat16
  explicit __HOST_DEVICE__ hip_bfloat16(float f) : data(float_to_bfloat16(f)) {}

  explicit __HOST_DEVICE__ hip_bfloat16(float f, truncate_t)
      : data(truncate_float_to_bfloat16(f)) {}

  __HOST_DEVICE__ operator float() const { … }
  …
};
```

C does not allow `enum`-inside-struct, default member functions,
member initializer lists, conversion operators, or
`= default;` — so the file cannot be parsed by a C compiler.

The same chain also pulls in C++ standard-library headers
(`<ostream>` from `amd_hip_bfloat16.h`; `<climits>` and `<cmath>`
from the related `amd_hip_bf16.h`) — each gated only on
`!defined(__HIPCC_RTC__)`, not on `__cplusplus`. So a plain C
compile fails on at least one of these long before reaching the
struct definition.

## Reproducer

```sh
$ cat > /tmp/test_c.c <<EOF
#define __HIP_PLATFORM_AMD__
#include <hipsparselt/hipsparselt.h>
int main(void) { return 0; }
EOF
$ gcc -I/opt/rocm/include -c /tmp/test_c.c -o /tmp/test_c.o
/opt/rocm/include/hip/amd_detail/amd_hip_bf16.h:107:10: fatal error:
    climits: No such file or directory
  107 | #include <climits>
      |          ^~~~~~~~~
compilation terminated.
```

If `<climits>` is replaced with `<limits.h>` and `<cmath>` with
`<math.h>` to get past the stdlib hurdle, the next failure is
inside the struct:

```
/opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h:43:42:
    error: expected ';', identifier, or '(' before '=' token
   43 |   __HOST_DEVICE__ hip_bfloat16() = default;
      |                                  ^
```

## Comparison with sibling libraries

`hipBLAS` solves the same `hip_bfloat16` aliasing requirement
correctly — it gates the C++-only typedef path behind a
preprocessor switch and falls back to a C-compatible POD struct:

```c
// /opt/rocm/include/hipblas/hipblas.h:5614-5634
#if defined(HIPBLAS_USE_HIP_BFLOAT16)
    #include <hip/hip_bfloat16.h>
    typedef hip_bfloat16 hipblasBfloat16;
#elif __cplusplus < 201103L || !defined(HIPBLAS_BFLOAT16_CLASS)
    /* minimal POD definition for plain-C consumers */
    typedef struct hipblasBfloat16 {
        uint16_t data;
    } hipblasBfloat16;
#else
    class hipblasBfloat16 { … };
#endif
```

`hipsparselt.h` has no equivalent fallback — it always pulls in
the C++-only definition.

## Root-cause split

This bug has two contributing layers:

1. **`hipsparselt.h` issue (this report)**: the unconditional
   `#include <hip/hip_bfloat16.h>` with no C-fallback typedef.
   Fixable without touching HIP itself.

2. **`hip/amd_detail/amd_hip_bfloat16.h` issue (separate, in
   ROCm/HIP)**: a header named `hip_bfloat16.h` (no language
   marker in its name and reachable from any HIP user code)
   defines a C++-only struct without a C-compatible spelling
   under `#ifndef __cplusplus`. Even consumers that try to be
   careful about C++ stdlib have no way to use the type from C.
   This is HIP's responsibility but worth flagging in this
   report because hipSPARSELt sits on top of it.

Either fix in isolation would unblock C consumers of
hipSPARSELt. Fix #1 is strictly local to this repo.

## Impact

Same as for the companion
[`hipblaslt_c_api_requires_cxx_compile.md`](hipblaslt_c_api_requires_cxx_compile.md):
any project that needs to consume hipSPARSELt as a C API is
blocked. Concrete cases:

- **Binding generators** parsing C headers with libclang in
  C-mode (`-x c`) or compiling C wrappers — Python / Rust / Go
  / OCaml / Lua wrappers all hit this.
- **Embedding via FFI in C-only host languages** (classic C,
  embedded toolchains without a linked C++ runtime).
- **Build systems that treat ROCm headers as a pure-C dependency**
  on the assumption that the `extern "C"` wrapper makes them so.

Sibling libraries that *do* parse cleanly as C in 7.13.0:
hipBLAS, hipFFT, hipRAND, hipSOLVER, hipSPARSE, hipDNN, RCCL,
roctracer, AMD COMGR. hipSPARSELt is the outlier.

## Suggested fix

Mirror the hipBLAS pattern in `hipsparselt.h` — gate the
`hip_bfloat16` import behind a switch and provide a POD fallback:

```c
// hipsparselt.h
#if defined(__HIP_PLATFORM_AMD__)
    #if defined(HIPSPARSELT_USE_HIP_BFLOAT16)
        #include <hip/hip_bfloat16.h>
    #elif __cplusplus < 201103L || !defined(HIPSPARSELT_BFLOAT16_CLASS)
        /* POD fallback for C compile */
        typedef struct hip_bfloat16 {
            uint16_t data;
        } hip_bfloat16;
    #else
        #include <hip/hip_bfloat16.h>
    #endif
#elif defined(__HIP_PLATFORM_NVIDIA__)
    typedef __nv_bfloat16 hip_bfloat16;
#endif
```

C consumers get a workable POD; C++ consumers (default) keep the
full feature-rich type when they opt in via the macro. The
function-signature-level uses of `hip_bfloat16 *` work uniformly
under either fallback because both share the same
`{ uint16_t data; }` layout.

(Long-term, the cleaner fix is upstream in HIP: split
`amd_hip_bfloat16.h` into a C-compatible POD struct definition
and a separate C++ helper-method header. But that is HIP's call
to make.)

## Workaround for downstream

Patch the header on the downstream side — replace the
`#include <hip/hip_bfloat16.h>` with an inline POD struct
definition. This is invasive enough that no production
downstream does it; instead, the standard workaround is to bind
hipSPARSELt at the C++ source level, paying the cost of pulling
in libstdc++.

## See also

The companion bug report
[`hipblaslt_c_api_requires_cxx_compile.md`](hipblaslt_c_api_requires_cxx_compile.md)
covers the same shape of problem for `hipblaslt.h`, plus an
additional hipBLASLt-specific issue (three unused C++ stdlib
includes — `<memory>`, `<regex>`, `<vector>` — at the top of
`hipblaslt.h`).
