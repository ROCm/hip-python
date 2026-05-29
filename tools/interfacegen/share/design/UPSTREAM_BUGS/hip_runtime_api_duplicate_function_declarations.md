# Bug report: `hip_runtime_api.h` declares `hipLaunchKernelExC` and `hipDrvLaunchKernelEx` more than once

**Filed against:** [`ROCm/HIP`](https://github.com/ROCm/HIP) for `hip_runtime_api.h`

**Encountered on:** 2026-05-28

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version` reports
`7.13.0`; all line numbers below are from this checkout).

**Affected file:** `/opt/rocm/include/hip/hip_runtime_api.h`.

## Title

`hipLaunchKernelExC` is declared twice and `hipDrvLaunchKernelEx`
three times in the same public header, in three different doxygen
group contexts.

## Summary

The public HIP runtime header re-declares two `Execution`-group
entry points at multiple physical locations inside the single
top-level `extern "C"` block:

```text
$ grep -n 'hipDrvLaunchKernelEx\|hipLaunchKernelExC' \
      /opt/rocm/include/hip/hip_runtime_api.h
6954: hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, …);
6970: hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, …);
9469: hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, …);
9867: hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, …);
9883: hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, …);
```

- `hipLaunchKernelExC` — 2 declarations (lines 6954, 9867).
- `hipDrvLaunchKernelEx` — 3 declarations (lines 6970, 9469, 9883).

Every copy is type-identical (same return type, same parameter
list, same `const` qualifiers, same parameter names). All five
declarations carry the doxygen tag `@ingroup Execution`, but they
are physically inside three different surrounding `@defgroup`
blocks:

| Lines        | Surrounding `@defgroup` (physical location) | `@ingroup` tag on the decl |
|--------------|---------------------------------------------|----------------------------|
| 6954, 6970   | `Execution` (opens before 6932, closes at 6991) | `Execution` |
| 9469         | `Graph` (opens at 8115, closes at 9474)         | `Execution` |
| 9867, 9883   | `Surface` (opens at 9787, closes at 9886)       | `Execution` |

This shape is consistent with someone duplicating the declaration
each time they wanted the symbol to appear inside another doxygen
group's textual range, instead of relying on a single declaration
with the appropriate `@ingroup` tag.

## Why it matters

### 1. Legal C, but a maintenance smell

C permits any number of compatible re-declarations of the same
function (C17 §6.7p4 and §6.2.7), so the compiler accepts the
header without complaint. That is also the failure mode: the
language gives no signal that the header carries three sources of
truth for the same symbol.

If a future patch changes the parameter list, the `const`
qualifiers, or the parameter names at one of the five sites but
not the others, the result will be either a hard compile-time
"conflicting types" error (best case — gets caught the moment a
TU includes the header) or, more insidiously, a documentation /
ABI drift where the doxygen output and the binding generators
disagree about the canonical signature.

### 2. Any tooling that walks function declarations sees them
all

Tools that consume the header via the Clang AST (libclang
binding generators, static analyzers, signature differs, ABI
checkers, swagger-style API documentation walkers, IDE indexers)
see one `FUNCTION_DECL` cursor per physical declaration. None of
them dedup by default — the Clang AST deliberately preserves
every declaration site so tooling can map back to file/line.

Downstream consumers therefore need either (a) their own
ad-hoc dedup pass per affected symbol, or (b) to assume the
header is well-formed and break when ROCm decides to duplicate
a third symbol. Neither is good. The fix belongs on the header
side because that is the single source of truth.

### 3. Doxygen output is unnecessarily noisy

Doxygen renders each declaration site separately; the generated
HTML reference therefore shows two entries for
`hipLaunchKernelExC` and three for `hipDrvLaunchKernelEx`, all
identical. Users encounter the same prototype several times
under the `Execution` group with no indication of which one is
authoritative.

## Suggested fix

Keep exactly one declaration per function — the original copies
in the `Execution` group at lines 6954 / 6970 — and delete the
four duplicates at lines 9469, 9867, and 9883.

If the original motivation for the duplicates was cross-group
visibility in the rendered doxygen output, the same effect is
achievable with doxygen's `@ingroup` mechanism on the single
canonical declaration (which is already what the existing
`@ingroup Execution` tag is for).

## Reproduction

```sh
# Show the duplicates:
grep -n 'hipDrvLaunchKernelEx\|hipLaunchKernelExC' \
    /opt/rocm/include/hip/hip_runtime_api.h

# Confirm via the preprocessor that the C compiler sees N
# declarations of each symbol after macro expansion:
echo '#include <hip/hip_runtime_api.h>' \
  | cc -E -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -xc - \
  | grep -cE '\bhipDrvLaunchKernelEx\b *\('
# expect: 3
echo '#include <hip/hip_runtime_api.h>' \
  | cc -E -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -xc - \
  | grep -cE '\bhipLaunchKernelExC\b *\('
# expect: 2
```

## Scope

This bug report covers only the two `Execution`-group entry
points known to be affected in ROCm 7.13.0. A broader audit of
`hip_runtime_api.h` for the same pattern is advisable but out of
scope here. A starting point:

```sh
# List every public function whose name appears more than once
# inside hip_runtime_api.h as a declarator (ignoring
# comments / @see / @brief mentions):
awk '/^[a-zA-Z_].*\(/{print}' \
    /opt/rocm/include/hip/hip_runtime_api.h \
  | grep -oE '\<hip[A-Z][A-Za-z0-9_]*\(' \
  | sort | uniq -c | awk '$1 > 1'
```
