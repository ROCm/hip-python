# Changelog

## Unreleased

### Per-header `nogil` for the LLVM recipe

`llvm_c` in `support/recipes/rocm.py` gained `nogil_headers`,
`is_nogil_header()` and a `nogil_node_init(header_relpath)` factory.
The factory marks a header's functions `nogil` in
`modifiers_lazy_loader` — which is what selects the with-nogil emitter
— and picks the exception sentinel from the return type: `except? NULL`
for pointers, `except? -1` for integral and boolean returns, `except?
<Enum>-1` for enums, and `noexcept` where nothing is left out of band
(`void`, records by value, floats). Functions taking a callback are
skipped.

The enum sentinel is a cast rather than a named constant because
`llvm-c` defines none: of its 43 named enums, not one declares a
negative enumerator, and the constants that read like an error marker
(`LLVMDSError`, `LLVMModuleFlagBehaviorError`, `LLVMCodeGenLevelNone`)
are values their getter returns in normal operation, so naming one
would put a GIL-taking `PyErr_Occurred` check on the common path. An
enum that does declare -1 falls back to `noexcept nogil` instead of
mistaking a valid return for a failed symbol load.

### `nogil` retval holder keeps a pointee's `const`

The with-nogil emitter declared the retval holder with all `const`
stripped, so a `const char *` return produced `cdef char * _cy_..
._retval` and Cython warned that the assignment inside the block
discards the qualifier — 14 warnings in the LLVM bindings once those
modules started releasing the GIL. The qualifier is now kept for
pointer returns, where it belongs to the pointee and leaves the local
assignable; a `const` *value* return still drops it, since Cython
rejects the assignment into a `cdef const T` local.

### Retired commit-count versioning

Removed the commit-count machinery from `support/gitversion.py`
(`version()`, `git_branch_rev_count()`, `__MAIN_BRANCH__`, and the
upstream/local-distance helpers), keeping only `git_rev()`,
`git_current_branch()`, `git_describe()`, and `git_is_clean()`. The
`support._base.versions()` helper no longer appends a commit count,
and `support.cython.write_version_py_in()` no longer emits the
count-composed `LONG_VERSION`/`_CODEGEN_VERSION`.

## \*.\*.\*.57.\* (2026-05-26)

**Scope.** Summarizes everything on
`dev/docharri/codegen-consolidation-interfacegen` not yet on
`origin/amd-integration` (57 commits). The 4th version slot is
the commit-count vs `amd-integration`; the other slots are
placeholders until the broader version scheme lands.

### Unified hip-python recipe

The per-library recipe forest collapsed into a single
pip-installable `recipes/hip-python/` tool. It emits directly
into `packages/<wheel>/src/` to match the new src-layout in
the consumer repo (see `hip-python/CHANGELOG.md`). Cross-module
imports, prefix-case handling, and per-library status nodes
are now generator-managed instead of recipe-managed.

### New library bindings emitted

The generator now produces wheels for **hipblaslt**,
**hipsparselt**, **hiptensor**, **hipdnn_backend**, **hsa**, and
**amdsmi** (including the ESMI CPU-monitoring block — 75
functions). Each binding came with its own pile of fixes:
prefix/case overrides, header-include workarounds (e.g.
`hsa_ext_finalize.h` folded via an `-include` cflag,
`hipfftXt.h` deliberately excluded), and hardcoded
INOUT-mistag overrides for caller-allocated buffers (`rccl`,
`hsa`, `hiptensor`, plus pointer-intent fixes for HIP and
amdsmi doxygen mistags). The amdsmi recipe enables the ESMI
block. The hipdnn_backend recipe was repaired to use the install-dir
layout and map `constexpr`.

### Cython generator features

Per-module output expanded from `.pxd`+`.pyx` to also include a
`.pyi` type-stub (with arg names and pyx docstrings, so
sphinx-autoapi can render the API without the compiled wheel)
and a Sphinx `.rst` wrapper. Generated `cy*` call sites now
emit `with nogil:`. Other additions: transitive Record admit,
structured `CallArgHoist`, transitive `AnonymousFunctionPointer`
ctypedef emission, foreign-record pointer parameters routed
through `Pointer`, C signature rendered as `.. rubric::` +
`code-block:: c` instead of a plain backticked line, C `const`
stripped from `:py:obj:` type references, `has_symbol` probe
codegen, and a Cython-3.0.x `*const *` codegen workaround.

### Doxygen / docstring quality

Docstring rendering received a series of correctness passes:
transitive `\copydoc` resolution at the cleaned-comment
chokepoint (so chains of `\copydoc` finally produce content),
group-priority hierarchy with multi-id `@ingroup` support,
pattern-C recovery of doc comments hidden behind `#if` guards,
per-file token walk that handles pattern A/B docs inside `@{`
blocks, brief inference from the first sentence, leaked-tag
cleanup, elision of the promoted brief from the details body
to avoid duplication, and a group-fallback path for orphaned
items.

### Generator refactor

The 3859-line Cython monolith was split into a 7-file package
(pure refactor — no behaviour change in that commit). The
codebase was renamed `hip` → `rocm` end-to-end to match the
target namespace. The `.pyi` renderer was refactored onto the
node hierarchy. Type rendering was routed through a single
`TypeHandler`, fixing inconsistent handling of typedef-of-typedef
chains, `EXTVECTOR` / `UNEXPOSED` / unknown `TypeKind` ids, and
header-tuple extraction edge cases.

### Upstream compatibility

The generator now tolerates libclang 17+ AST shapes and
pyparsing 3.0+ APIs (previously pinned to older versions). The
Cython floor moved to `>=3.1.0` for the 3.0.x `*const *` codegen
bug (see `share/design/BUILDING.md` under "Cython version
requirement"). Upstream-bug notes were recorded for hipblaslt +
hipsparselt C-includability issues, HIP / amdsmi doxygen
pointer-intent mistags, and the in-depth hipblaslt follow-up —
the workarounds live in the per-library recipes.
Always-consulted source trees
(rocm-systems, llvm-project) now produce a generator warning
when their commit hash can't be recorded.

### Codegen artifact pipeline

The generator now writes a `generated_versions.cmake` per
package, capturing upstream commit hashes (rocm-libraries,
rocm-systems, llvm-project) and the codegen date — this is
what the consumer repo's docs landing page substitutes from.
Header workarounds are persisted to disk for reproducibility.
`docs_src/sphinx/_toc.yml.in.in` is rendered to
`_toc.yml.in` with per-subtree module lists. `.pyi`/`.rst`
files are registered as generator-owned in the module
docstring so editors and reviewers don't mistake them for
hand-edited.

---

**Cross-reference.** The wheel build, runtime, and docs
machinery that consumes the artifacts emitted by this
generator lives in **hip-python** — see that repo's
`CHANGELOG.md` for the build / runtime / docs changes.
