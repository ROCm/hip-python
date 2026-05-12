# Bug report: full inventory of C++-only constructs in `hipblaslt`'s public C-API headers

**Filed against:** [`ROCm/rocBLAS-internal` →
hipBLASLt](https://github.com/ROCm/rocBLAS-internal)

**Encountered on:** 2026-05-12

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version`
reports `7.13.0`; line numbers below are from this checkout).

**Affected files:**
- `/opt/rocm/include/hipblaslt/hipblaslt.h` (1022 lines, the
  umbrella public-API header)
- `/opt/rocm/include/hipblaslt/hipblaslt-types.h` (transitively
  included)
- `/opt/rocm/include/hipblaslt/hipblaslt_e8.h`,
  `hipblaslt_e5m3.h`, `hipblaslt_bfloat6.h`,
  `hipblaslt_float6.h`, `hipblaslt_float4.h` (transitively
  included via `hipblaslt-types.h`)

**Companion to:**
[`hipblaslt_c_api_requires_cxx_compile.md`](hipblaslt_c_api_requires_cxx_compile.md),
which covered the `<memory>` / `<regex>` / `<vector>` /
`<hip/hip_bfloat16.h>` include-strip workarounds. The original
report's "the public API is C-callable; just remove the unused
includes" framing turns out to be incomplete: even with all the
problematic includes stripped, the public API headers themselves
still contain C++-only syntactic constructs that prevent a plain
C compiler from parsing them. This report enumerates every such
construct discovered by an exhaustive scan.

## Title

`hipblaslt`'s public C-API headers contain **five distinct
categories** of C++-only syntax — beyond the unused stdlib
includes already covered in the prior report. Together they
make the headers fundamentally non-includable from a C
translation unit even after every workaround a downstream can
apply via include-strip patches.

## Summary table

| # | Category | Severity | File:line |
|---|---|---|---|
| 1 | C++ stdlib includes | high (breaks `#include`) | `hipblaslt.h:54-56` (covered in prior report) |
| 2 | C++-only `hip_bfloat16` via include | high | `hipblaslt.h:58` (covered in prior report) |
| 3 | **Default member initializers in a public struct definition** | high — blocks any include-strip workaround | `hipblaslt.h:410-416` |
| 4 | `static_cast<T>(...)` and `static_assert(false, ...)` in public macros | medium — only fires on macro expansion | `hipblaslt.h:71-80` |
| 5 | Pure-C++ transitively-included extension headers (no C fallback) | high — propagates upward through `hipblaslt-types.h` | `hipblaslt_e8.h`, `hipblaslt_e5m3.h`, `hipblaslt_bfloat6.h`, `hipblaslt_float6.h`, `hipblaslt_float4.h` |

Sections 1 and 2 were the subject of the prior bug report.
Sections 3, 4 and 5 are the focus of this follow-up.

---

## Category 3: Default member initializers in `_hipblasLtMatmulHeuristicResult_t`

`hipblaslt.h:410-416`:

```c
typedef struct _hipblasLtMatmulHeuristicResult_t{
  hipblasLtMatmulAlgo_t algo;                      /**<Algo struct*/
  size_t workspaceSize = 0;                        /**<Actual size of workspace memory required.*/
  hipblasStatus_t state = HIPBLAS_STATUS_SUCCESS;  /**<Result status. ...*/
  float wavesCount = 1.0;                          /**<Waves count is a device utilization metric. ...*/
  int reserved[4];                                 /**<Reserved.*/
} hipblasLtMatmulHeuristicResult_t;
```

C does not allow assignment-syntax default member initializers
inside a struct declaration. gcc rejects the file unconditionally
in C mode:

```
hipblaslt.h:412:24: error: expected ':', ',', ';', '}' or
    '__attribute__' before '=' token
  412 |   size_t workspaceSize = 0;
      |                        ^
```

This is the construct that defeats every include-strip
workaround a downstream can apply. The struct sits in the
public API surface (it's the result type of
`hipblasLtMatmulAlgoGetHeuristic`), so it cannot be omitted —
a binding generator that wants to expose the function MUST
parse this struct.

### Compare with the well-formed sibling struct

The IMMEDIATELY ADJACENT struct on the same page does it
right (`hipblaslt.h:389-396`):

```c
typedef struct _hipblasLtMatmulAlgo_t{
#ifdef __cplusplus
  uint8_t data[16] = {0};
  size_t max_workspace_bytes = 0;
#else
  uint8_t data[16];
  size_t max_workspace_bytes;
#endif
} hipblasLtMatmulAlgo_t;
```

— guarded by `#ifdef __cplusplus` with a plain-C fallback.
The author clearly knew the C++ default-initializer syntax
would break C consumers. The fix for
`_hipblasLtMatmulHeuristicResult_t` is to apply the same
pattern. **Roughly five-line diff against upstream.**

### Suggested fix

```diff
 typedef struct _hipblasLtMatmulHeuristicResult_t{
   hipblasLtMatmulAlgo_t algo;
+#ifdef __cplusplus
   size_t workspaceSize = 0;
   hipblasStatus_t state = HIPBLAS_STATUS_SUCCESS;
   float wavesCount = 1.0;
+#else
+  size_t workspaceSize;
+  hipblasStatus_t state;
+  float wavesCount;
+#endif
   int reserved[4];
 } hipblasLtMatmulHeuristicResult_t;
```

C consumers initialise the fields after construction (e.g. via
`memset(&result, 0, sizeof(result));` or per-field assignment)
— the C++ default-initialisation behaviour is preserved exactly
for C++ consumers, and zero behavioural change for ABI.

---

## Category 4: `static_cast<>` and `static_assert(false, ...)` in public macros

`hipblaslt.h:71-80`:

```c
#define HIPBLASLT_DATATYPE_INVALID static_cast<hipDataType>(255)
#define HIPBLASLT_COMPUTE_TYPE_INVALID static_cast<hipblasComputeType_t>(0)
#define HIPBLASLT_OPERATION_INVALID static_cast<hipblasOperation_t>(0)
#define ROCBLASLT_COMPUTE_TYPE_INVALID static_cast<rocblaslt_compute_type>(255)

#define HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT \
    static_assert(false, "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT is deprecated and not supported. ...")
#define HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT \
    static_assert(false, "HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT is deprecated and not supported. ...")
```

`static_cast<T>(value)` is C++ syntax. C uses the cast operator
`((T)(value))`. `static_assert(condition, "msg")` was added to C
only in C11 (as `_Static_assert`) and given the
`static_assert` spelling in C23 — and even so, the form
`static_assert(false, "...")` (deprecated-marker pattern) is
unusual.

These don't break a plain `#include` because macros are only
evaluated at use sites. But:

- Any C consumer that happens to USE `HIPBLASLT_DATATYPE_INVALID`
  will fail to compile.
- A binding generator that emits these macros as Python
  constants needs to handle the fact that `static_cast<>` isn't
  a C expression.

### Suggested fix

For the `*_INVALID` casts, switch to a C-compatible cast that
also remains valid C++:

```diff
-#define HIPBLASLT_DATATYPE_INVALID static_cast<hipDataType>(255)
+#define HIPBLASLT_DATATYPE_INVALID ((hipDataType)255)
-#define HIPBLASLT_COMPUTE_TYPE_INVALID static_cast<hipblasComputeType_t>(0)
+#define HIPBLASLT_COMPUTE_TYPE_INVALID ((hipblasComputeType_t)0)
-#define HIPBLASLT_OPERATION_INVALID static_cast<hipblasOperation_t>(0)
+#define HIPBLASLT_OPERATION_INVALID ((hipblasOperation_t)0)
-#define ROCBLASLT_COMPUTE_TYPE_INVALID static_cast<rocblaslt_compute_type>(255)
+#define ROCBLASLT_COMPUTE_TYPE_INVALID ((rocblaslt_compute_type)255)
```

For the `static_assert(false, ...)` deprecated-marker macros:
either drop the macros entirely (the deprecation message can
move to a comment), or use a portable spelling like
`#error "..."` would for compile-time failure under both C and
C++. But the simplest fix is to remove the macros — the user
seeing `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT` in
release-notes searches gets the same message via a comment in
the header.

---

## Category 5: Pure-C++ extension headers transitively pulled in via `hipblaslt-types.h`

`hipblaslt-types.h` (lines 25-31) includes:

```c
#include <hip/hip_fp8.h>
#include "hipblaslt_float8.h"
#include "hipblaslt_bfloat6.h"
#include "hipblaslt_float6.h"
#include "hipblaslt_float4.h"
#include "hipblaslt_e8.h"
#include "hipblaslt_e5m3.h"
```

Per-header C-fallback survey (built via grep for
`__cplusplus < 201103L` and structural keywords):

| Header | Has `__cplusplus` C-fallback? | C++ classes / structs | Operator overloads | `= default` constructors |
|---|---|---|---|---|
| `hipblaslt_e8.h` | NO | 1 | 1 | 1 |
| `hipblaslt_e5m3.h` | NO | 1 | 1 | 2 |
| `hipblaslt_bfloat6.h` | NO | 2 | 2 | 1 |
| `hipblaslt_float6.h` | NO | 2 | 2 | 1 |
| `hipblaslt_float4.h` | NO | 1 | 2 | 1 |
| `hipblaslt_float8.h` | YES (3 conditional blocks) | 4 | 15 | 0 |

`hipblaslt_float8.h` is the model for how the others SHOULD be
structured: a `#if __cplusplus < 201103L || (!defined(__HCC__)
&& !defined(__HIPCC__))` branch that defines the type as a POD
`typedef struct { uint8_t __x; } NAME;` for plain-C consumers,
plus a C++ class in the `#else` branch. The other five
extension headers have no equivalent fallback — they are pure
C++.

### Sample (from `hipblaslt_e8.h`):

```cpp
struct HIPBLASLT_EXPORT hipblaslt_e8 {
    uint8_t data;

    // default constructor
    HIP_HOST_DEVICE hipblaslt_e8() = default;

    HIP_HOST_DEVICE hipblaslt_e8(float v0)
    {
        union {
            uint32_t x;
            float f;
        } v;
        v.f = v0;
        // ...
    }

    HIP_HOST_DEVICE explicit operator float() const {
        // ...
    }
};
```

Constructors, conversion operators, `= default` member
function specifiers — none of which are C.

### Are they actually USED?

Importantly: a `grep` over `hipblaslt.h` itself (the umbrella
public-API header) shows **zero references** to any of
`hipblaslt_e8`, `hipblaslt_e5m3`, `hipblaslt_bfloat6`,
`hipblaslt_float6`, `hipblaslt_float4`, `hipblaslt_f8`, or
`hipblaslt_bf8`. The public-API surface uses opaque handles
and `hipDataType` / `hipblasComputeType_t` enums to address
these formats — never the struct types directly.

So the cost of these C++ headers being unconditionally pulled
in by `hipblaslt-types.h` is paid by every C consumer for **no
public-API benefit**. The types exist for hipblaslt's own
internal kernels and the hipblaslt-ext.hpp C++ extension
surface; they don't need to be visible from `hipblaslt.h`.

### Suggested fix

Either:

- (a) **Add the same `__cplusplus` fallback** to each of the
  five headers (mirror `hipblaslt_float8.h`'s pattern). Five
  small per-file diffs of ~20 lines each.

- (b) **Move the includes out of `hipblaslt-types.h`** and into
  a C++-only sub-header (e.g. a new `hipblaslt-extension-types.hpp`
  that's `#include`d only from `hipblaslt-ext.hpp`). The C-API
  `hipblaslt-types.h` then contains only the POD typedefs
  (`hipblasLtFloat`, `hipblasLtHalf`, `hipblasLtBfloat16`,
  `hipblasLtInt8`, `hipblasLtInt32`) it currently does. **One
  diff in `hipblaslt-types.h` plus the new file.**

(b) is the cleaner separation of concerns — a C-API header
containing only C-compatible declarations, an explicitly C++-only
sibling for the extension types.

---

## Cumulative impact

For a downstream binding generator that wants to consume
hipBLASLt as a C API, fixing categories 1+2 (already filed)
gets `hipblaslt.h` parseable through libclang's `-x c` mode
(with recovery — libclang is more forgiving than gcc). But the
**gcc compile of any binding glue that `#include`s
`hipblaslt.h`** still fails because of:

- Category 3 (the default-initializer struct in the umbrella
  header itself).
- Category 5 (the chain of pure-C++ extension headers pulled
  in via `hipblaslt-types.h`).

Working around 3 + 5 downstream requires either:

- Patching the upstream `hipblaslt-types.h` to comment out the
  five extension-type includes AND patching `hipblaslt.h` to
  rewrite the default initializers — invasive enough that no
  shipping consumer is doing it.
- Switching the binding pipeline to compile as C++ — adds
  libstdc++ as a runtime link dependency, defeats the purpose
  of the `extern "C"` wrapper.

All five categories combined make hipBLASLt the **only ROCm
math library binding that cannot be built in plain C** as of
ROCm 7.13.0. The sibling libraries hipBLAS, hipFFT, hipRAND,
hipSOLVER, hipSPARSE, hipDNN, plus the related hipSPARSELt
(after the fixes from the companion bug report) all parse and
compile cleanly under a C-only toolchain.

## Recommended fix priority

1. **Category 3 (5-line diff).** Single-line change to add
   `#ifdef __cplusplus` guard around three field initializers.
   Highest ROI; unblocks every other workaround.
2. **Category 5 option (b)** (move extension types to a
   separate header). Cleanest architecturally; matches what
   `hipblaslt-ext.hpp` (the C++ extension API) is already doing.
3. **Categories 1 + 2** (already in the prior report).
4. **Category 4** (static_cast/static_assert in macros). Only
   bites users who actually invoke those specific macros.

## Workaround for downstream

Until 1+5 land upstream, downstream bindings have to:

- Either compile their glue as C++ (pulls in libstdc++).
- Or ship a wholesale-rewritten copy of `hipblaslt.h` and
  `hipblaslt-types.h` in the binding's source tree, accept the
  maintenance burden of keeping it in sync with each ROCm
  release, and patch every problematic struct + macro by hand.

Neither option is a comfortable place for shipping software.
The `hip-python` project currently ships `hipsparselt`
(via the include-strip pipeline documented in
`share/design/UPSTREAM_BUGS/hipsparselt_c_api_requires_cxx_compile.md`)
but **not hipblaslt** — exactly because category 3 above
defeats every workaround the codegen can apply locally.
