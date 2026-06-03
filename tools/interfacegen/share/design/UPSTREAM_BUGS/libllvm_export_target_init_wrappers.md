# Defect report: ROCm `libLLVM.so` is missing the public LLVM-C target-initialization entry points

**Filed against:** ROCm Compiler (the LLVM toolchain shipped in the ROCm
SDK; `ROCm/llvm-project`), component `libLLVM.so`.

**Issue type:** Defect — the shipped shared library's exported symbol set
is incomplete relative to the documented public LLVM-C API.

**Severity:** Major. Documented public LLVM-C entry points are uncallable
from the shipped `libLLVM.so`; the only workaround is to rebuild the
library from static archives.

**Encountered on:** 2026-06-02.

**Affected ROCm version:** 7.13.0 (`/opt/rocm/.info/version` reports
`7.13.0`). The shared library is
`/opt/rocm/llvm/lib/libLLVM.so -> libLLVM.so.23.0git` (130 MB).

**Affected consumers:** `hip-python` (`rocm.bindings.llvm.*`) and, on top
of it, `numba-hip`. Any tool that consumes the LLVM-C API by **dynamic
symbol name** (Python/ctypes/dlopen, or bindings for other non-C
languages) hits the defect identically.

> **Origin and ownership (please read first).** The *root cause* is a
> design decision in **upstream LLVM**: the affected entry points are
> declared `static inline` in `<llvm-c/Target.h>`, so by design no shared
> library ever exports them. Revisiting that upstream decision (so the
> LLVM-C target-init entry points become real exported symbols) is
> probably worth pursuing on its own merits, since it would benefit every
> by-name LLVM-C consumer, not just ROCm's. That is a longer path,
> though — so **in the meantime an alternative approach can be explored
> entirely within the ROCm packaging of `libLLVM.so`**: compile and
> export a small de-inlined shim (see [Suggested fix](#suggested-fix)).
> This needs no upstream LLVM change and has no ABI impact on existing
> symbols. In other words: upstream's design causes it; an upstream fix
> is desirable but slow, and ROCm packaging can resolve it now.

## Summary

The LLVM-C API documents `LLVMInitializeAll*` / `LLVMInitializeNative*`
as *the* public entry points for target initialization. The
ROCm-distributed `libLLVM.so` does **not** export externally-linkable
definitions of them: ten such functions are absent from the library's
dynamic symbol table. Any consumer that binds the LLVM-C API by dynamic
symbol name (`dlopen`/`dlsym`, `GetProcAddress`) therefore cannot call
documented public API functions, even though the per-target initializers
those functions wrap *are* exported by the same library.

The ten missing symbols are the convenience target-initialization
wrappers declared `static inline` in `<llvm-c/Target.h>`:

| symbol | header (`/opt/rocm/llvm/include/llvm-c/Target.h`) |
|---|---|
| `LLVMInitializeAllTargetInfos`     | `static inline` (line 79)  |
| `LLVMInitializeAllTargets`         | `static inline` (line 88)  |
| `LLVMInitializeAllTargetMCs`       | `static inline` (line 97)  |
| `LLVMInitializeAllAsmPrinters`     | `static inline` (line 106) |
| `LLVMInitializeAllAsmParsers`      | `static inline` (line 115) |
| `LLVMInitializeAllDisassemblers`   | `static inline` (line 124) |
| `LLVMInitializeNativeTarget`       | `static inline` (line 134) |
| `LLVMInitializeNativeAsmParser`    | `static inline` (line 149) |
| `LLVMInitializeNativeAsmPrinter`   | `static inline` (line 161) |
| `LLVMInitializeNativeDisassembler` | `static inline` (line 173) |

## Steps to reproduce

Standard tools only — no hip-python, no Python bindings, no wheel build.
This is the by-name (`dlopen`/`dlsym`) view that every non-C-language
LLVM-C consumer sees.

1. Inspect the dynamic symbol table:

```text
$ nm -D --defined-only /opt/rocm/llvm/lib/libLLVM.so \
    | grep -c LLVMInitializeAllTargetInfos      # -> 0  (missing wrapper)
$ nm -D --defined-only /opt/rocm/llvm/lib/libLLVM.so \
    | grep -c LLVMInitializeAMDGPUTargetInfo    # -> 1  (per-target init present)
```

2. Resolve the symbols at runtime exactly as a by-name consumer would:

```c
/* repro.c — build: cc repro.c -ldl -o repro && ./repro */
#include <dlfcn.h>
#include <stdio.h>

int main(void) {
  void *h = dlopen("/opt/rocm/llvm/lib/libLLVM.so", RTLD_NOW | RTLD_LOCAL);
  const char *names[] = {
      "LLVMInitializeAllTargetInfos",   /* static inline wrapper */
      "LLVMInitializeAMDGPUTargetInfo", /* per-target init       */
      (void *)0,
  };
  for (int i = 0; names[i]; ++i)
    printf("%-34s %s\n", names[i], dlsym(h, names[i]) ? "PRESENT" : "MISSING");
  return 0;
}
```

## Expected behavior

The documented public LLVM-C target-init entry points resolve from the
stock `libLLVM.so` and can be called:

```text
LLVMInitializeAllTargetInfos       PRESENT
LLVMInitializeAMDGPUTargetInfo     PRESENT
```

Equivalently, all ten wrappers appear in the dynamic symbol table:

```text
$ nm -D --defined-only /opt/rocm/llvm/lib/libLLVM.so \
    | grep -Ec 'LLVMInitialize(All|Native)'
10
```

## Actual behavior

The ten wrappers are absent from the dynamic symbol table; `dlsym`
returns `NULL` for each, so calling them is impossible without rebuilding
the library:

```text
LLVMInitializeAllTargetInfos       MISSING
LLVMInitializeAMDGPUTargetInfo     PRESENT

$ nm -D --defined-only /opt/rocm/llvm/lib/libLLVM.so \
    | grep -Ec 'LLVMInitialize(All|Native)'
0
```

A Python/dlopen consumer that imports the wrapper fails at symbol
resolution, e.g. `RuntimeError: failed to dlsym
'LLVMInitializeAllTargetInfos'`.

## Impact

Target initialization is mandatory before any LLVM codegen, disassembly,
or target-data-layout query, and `LLVMInitializeAll*` /
`LLVMInitializeNative*` are the documented way to do it. Because they are
unreachable from the shipped library, a by-name consumer must either:

- hand-roll the per-target initialization loop itself — brittle, and it
  duplicates upstream logic that changes with the configured target list;
  or
- rebuild `libLLVM.so` from the LLVM static archives.

Concrete fallout in the affected consumers:

- **`examples/2_Advanced/list_targets.py`** (hip-python) needs
  `LLVMInitializeAllTargetInfos`, `LLVMInitializeAllTargets`,
  `LLVMInitializeAllTargetMCs` to enumerate targets and query each
  target's data layout. It currently probes at runtime and **skips
  itself** when run against the stock system `libLLVM.so`.
- **`numba-hip`** drives the same LLVM-C bindings for AMDGPU device-code
  generation and fails at `dlsym` for these symbols against the stock
  library.

The following snippet — derived from numba-hip's `AMDGPUTargetMachine`
(`numba/hip/amdgcn.py`) — shows the canonical usage in HIP Python:
building the AMDGPU target machine and reading its data layout. The three
mandatory `LLVMInitializeAll*` calls at the top are exactly the symbols
missing from the stock `libLLVM.so`, so this code raises at `dlsym` time
on the very first line unless the workaround library is used.

```python
from rocm.bindings.llvm.c.core import LLVMDisposeMessage
from rocm.bindings.llvm.c.target import (
    LLVMInitializeAllTargetInfos,   # <-- MISSING from stock libLLVM.so
    LLVMInitializeAllTargets,       # <-- MISSING from stock libLLVM.so
    LLVMInitializeAllTargetMCs,     # <-- MISSING from stock libLLVM.so
)
from rocm.bindings.llvm.c.targetmachine import (
    LLVMCodeGenOptLevel,
    LLVMCodeModel,
    LLVMCopyStringRepOfTargetData,
    LLVMCreateTargetDataLayout,
    LLVMCreateTargetMachine,
    LLVMDisposeTargetMachine,
    LLVMGetTargetFromTriple,
    LLVMRelocMode,
)

# 1. Target initialization — REQUIRED before any target lookup. All three
#    are needed, and all three are absent from the stock libLLVM.so.
LLVMInitializeAllTargetInfos()
LLVMInitializeAllTargets()
LLVMInitializeAllTargetMCs()

# 2. Look up the AMDGPU target and build a target machine for an ISA.
triple = b"amdgcn-amd-amdhsa"
(status, target, error) = LLVMGetTargetFromTriple(triple)
if status:
    raise RuntimeError(str(error))
tm = LLVMCreateTargetMachine(
    target,
    triple,
    b"gfx90a",  # target CPU == AMD GPU arch
    b"",        # target features
    LLVMCodeGenOptLevel.LLVMCodeGenLevelDefault,
    LLVMRelocMode.LLVMRelocDefault,
    LLVMCodeModel.LLVMCodeModelDefault,
)

# 3. Query the data layout that numba-hip stamps onto every device module.
data_layout = LLVMCreateTargetDataLayout(tm)
data_layout_cstr = LLVMCopyStringRepOfTargetData(data_layout)
print("amdgcn-amd-amdhsa / gfx90a data layout:", data_layout_cstr.decode("utf-8"))
LLVMDisposeMessage(data_layout_cstr)
LLVMDisposeTargetMachine(tm)
```

To work around the defect, hip-python carries a build-time rebuild
(`packages/rocm-bindings-compiler/bundled/libllvm/`): a de-inlined copy
of `<llvm-c/Target.h>` (`Target.cpp`) that gives the ten wrappers
external linkage, linked into a self-contained `libLLVM.so` from the LLVM
static archives (`llvm-config --link-static --libfiles`) with
`--whole-archive` (`HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON`, used by CI in
`ci/internal/build-wheels.sh`). This workaround is expensive:

- it **requires the LLVM static archives to be present** —
  `llvm-config --link-static --libfiles` fails when the SDK ships only
  the shared `libLLVM`, in which case there is no way to obtain the
  symbols at all;
- it imposes a **much longer link step and a larger wheel** (re-linking a
  ~130 MB library from static archives on every build); and
- it forces hip-python to **vendor and maintain an LLVM upstream header**
  (`Target.cpp`) in sync across LLVM versions.

All of this exists solely to materialize ten one-line wrapper functions
that ROCm's own LLVM build could export directly.

## Root-cause analysis

The wrappers are declared `static inline` in `<llvm-c/Target.h>`, so the
compiler never emits an external definition for them in **any** shared
library — each C/C++ translation unit is expected to recompile the inline
body by `#include`-ing the header. A consumer that binds the C API by
symbol name has no translation unit in which to do that, so the functions
are simply unavailable to it.

Crucially, the wrappers only expand to a sequence of calls to the
per-target initializers (`LLVMInitialize<Target>TargetInfo`, etc.), and
those per-target symbols **are** exported by the same `libLLVM.so`. So the
defect is narrow and self-contained: the shipped artifact exports the
building blocks but not the documented public aggregates built from them.

(The ExecutionEngine link-in helpers `LLVMLinkInMCJIT` /
`LLVMLinkInInterpreter` are **present** in the library — they are noted
here only to scope the defect to the ten `static inline` target-init
wrappers and exclude the JIT components.)

## Suggested fix

Export non-inline definitions of the ten target-init wrappers from
`libLLVM.so`. ROCm's libLLVM build can compile a tiny de-inlined shim
(exactly what hip-python's `bundled/libllvm/Target.cpp` already does) and
add these symbols to the dylib's export set. This is a small, additive
change with no ABI risk to existing symbols, and it makes the documented
LLVM-C target-init entry points reachable by every by-name consumer.

**Definition of done:**

```text
$ nm -D --defined-only /opt/rocm/llvm/lib/libLLVM.so \
    | grep -Ec 'LLVMInitialize(All|Native)'
10
```

i.e. all ten wrappers are present in the dynamic symbol table of the
stock ROCm `libLLVM.so`, with no rebuild required.

## Validation (for the hip-python owner — not the ROCm Compiler team)

This section does **not** apply to the ROCm Compiler team and requires
hip-python; it records how the hip-python maintainer validates the
end-to-end fix on the Python stack once a corrected `libLLVM.so` ships.

Against the fixed *stock* system `libLLVM.so` (with
`HIP_PYTHON_FORCE_BUILD_LIBLLVM=OFF`, i.e. **not** the static-archive
aggregate):

```python
from rocm.bindings.llvm.c import target
assert target.has_symbol("LLVMInitializeAllTargetInfos")   # now present
from rocm.bindings.llvm.c.target import LLVMInitializeAllTargetInfos
LLVMInitializeAllTargetInfos()                             # resolves + runs
```

and the example runs without its `has_symbol(...)` skip:

```sh
python examples/2_Advanced/list_targets.py   # prints the target list, no skip
```

Once this passes, the `HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON` workaround can
be left off (the default) and the compiler wheel can simply rely on the
stock system `libLLVM.so`.

## References

- `packages/rocm-bindings-compiler/bundled/libllvm/CMakeLists.txt`
  (`HIP_PYTHON_FORCE_BUILD_LIBLLVM`, `--whole-archive`,
  `llvm-config --link-static --libfiles`).
- `packages/rocm-bindings-compiler/bundled/libllvm/Target.cpp` (the
  de-inlined wrapper shim).
- `packages/rocm-bindings-core/src/rocm/bindings/util/posixloader.pyx`
  (`dlsym`/`has_symbol` by-name loading).
- `examples/2_Advanced/list_targets.py` (consumer + the runtime skip).
- `ci/internal/build-wheels.sh` (CI force-builds libLLVM to get these
  symbols).
- Upstream header: `/opt/rocm/llvm/include/llvm-c/Target.h`.
