# Changelog

Entries are grouped by audience. **Bindings** is what changes in the
generated output, and so in what a user of the bindings sees; **Codegen**
is the generator itself, its recipes and its tooling. The build, runtime
and docs machinery that consumes these artifacts lives in **hip-python**
— see that repository's `CHANGELOG.md`.

## 0.5

The minor bump is a signal to consumers: stubs generated before and
after this version differ in what they declare, and the older ones are
missing names the modules bind.

### Fixed

#### Bindings

- **Handle typedefs are declared.** `hipStream_t`, `hipEvent_t`,
  `hipFunction_t` and the 51 further aliases `hip` binds as `X = Y` were
  absent from the stub, so the names callers actually write did not
  resolve. They render as the same assignment, and carry the members of
  the record they alias.
- **A wrapper class names the base it inherits.** Every record and
  function pointer derives from the util package's `Pointer` in the
  `.pyx` and from nothing in the stub, which dropped `createRef`,
  `is_ptr_null` and the rest of the handle surface.
- **A leading underscore no longer hides a class.** `_hiprtcProgram` is
  bound by its module and named by a handle typedef; only compiler
  predefines such as `__llvm__` are skipped now.
- **`has_symbol` is declared** by a runtime-linked module. It comes from
  the module prolog rather than from a node, so the stub walker never
  saw it.

## 0.4

The minor bump is a signal to consumers: stubs generated before and
after this version differ in what they declare, and the older ones are
what make a type checker reject working code.

### Fixed

#### Bindings

- **Enum constants and record fields are declared in the `.pyi`.** Every
  enum, record and function pointer used to render as a bare class with
  a placeholder `__init__`, so `hipError_t.hipSuccess` and
  `hipDeviceProp_t().gcnArchName` read as errors. Named enums now render
  as `enum.IntEnum` subclasses carrying their constants, and records
  declare the properties the `.pyx` renders — decided by the one
  generator `PROPERTIES()` is built from — plus the method set every
  record gets from the templates.
- **An anonymous enum stubs as its constants** rather than as a class
  the `.pyx` never binds. Its constants were missing from the stub
  entirely; `HIP_SUCCESS` is one of them.
- **Hoisted anonymous records carry the name the module exports.** The
  stub used the node's local name, which restarts at `struct_0` inside
  every parent, so unrelated types collided under one name and the 39
  real `<parent>_struct_0` names were absent.
- **A cuda enum alias no longer derives from the hip enum's private
  base.** No stub declares that base, so a checker resolved the class to
  Unknown and accepted any attribute on it. The alias renders as
  `enum.IntEnum` with the constants it binds, hip spellings and cuda
  aliases alike.

## 0.3

The first version under the real version scheme, so it carries
everything that came before it: the entries previously filed under
`*.*.*.57.*` are folded in here, that slot having been a commit count
standing in until this scheme landed.

The minor bump is a signal to consumers: bindings generated before and
after this version differ in the integer widths they pin and in which
LLVM wrappers release the GIL, so the two are not interchangeable.

### Added

#### Bindings

- **Bindings for hipBLASLt, hipSPARSELt, hipTensor, hipDNN backend and
  amdsmi**, the last including the ESMI CPU-monitoring block's 75
  functions. Each came with its own set of prefix and case overrides,
  header-include workarounds (`hipfftXt.h` is deliberately excluded) and
  pointer-intent overrides for caller-allocated buffers that doxygen
  mistags as `INOUT`.
- **A `.pyi` type stub and a Sphinx `.rst` wrapper per module**, beside
  the `.pxd` and `.pyx`. The stub carries argument names and the `.pyx`
  docstrings, so the API renders without the compiled wheel on
  `sys.path`. Both are marked generator-owned in their module docstring
  so nobody mistakes them for hand-edited files.
- **A non-raising `has_symbol` probe** in the generated output, so a
  caller can test for a missing entry point without exception flow.

#### Codegen

- **Freestanding `stddef.h`, `stdint.h` and `stdbool.h`**, reached for
  only when the caller names no `-resource-dir` — which every production
  path does. `stddef.h` comes from the compiler rather than the C
  library and the PyPI libclang wheel ships no resource directory to
  hold it, so a parse that mentioned `size_t` had clang quietly recover
  it to `int`. They are spelled through clang predefines, so the widths
  follow the parse target rather than the generating host.
- **Per-header `nogil` opt-in for the LLVM recipe**, through
  `llvm_c.nogil_headers` — per header rather than per library, which is
  new for the generator. The recipe picks each function's exception
  sentinel from its return type, so a missing libLLVM still surfaces at
  the first call without the caller holding the GIL to find out, and a
  function taking a callback keeps the GIL whatever it returns. An
  enum's sentinel is a cast rather than a named constant because
  `llvm-c` declares no negative enumerator in any of its 43 named enums.
- **A recipe can alias a module's width-carrying typedefs** to the
  `stdint` names. Both per-module maps are validated when the backend is
  built.
- **One pip-installable recipe.** The per-library recipe forest
  collapsed into a single tool that emits straight into
  `packages/<wheel>/src/`. Cross-module imports, prefix-case handling
  and per-library status nodes are generator-managed rather than
  recipe-managed.
- **A `generated_versions.cmake` per package**, carrying the upstream
  commit hashes (rocm-libraries, rocm-systems, llvm-project) and the
  codegen date that the consumer's docs landing page substitutes from.
  Header workarounds are persisted to disk for reproducibility, and
  `docs_src/sphinx/_toc.yml.in` is rendered with per-subtree module
  lists.

### Changed

#### Bindings

- **Fixed-width typedefs keep their name instead of canonicalising.**
  Canonicalising `uint64_t` yields `unsigned long` on an LP64 host and
  `unsigned long long` on Windows, so generated bindings baked in the
  data model of whichever machine ran the generator. Each such typedef
  now carries its signedness and width in `FIXED_WIDTH_INT_SPECS`: the
  renderer takes the spelling and the pointer handler takes the width,
  so a `.pxd` declaration and the `.pyx` body can no longer disagree
  about the width of a variable the callee writes to. The autoconversion
  table knows these typedefs are integers, so stubs annotate `int`
  rather than falling through to an opaque type.
- **The generated `cy*` call sites emit `with nogil:`**, so they can be
  called in parallel.
- **Docstrings are considerably better.** `\copydoc` resolves
  transitively, so chains finally produce content; `@ingroup` supports
  multiple ids under a group-priority hierarchy; doc comments hidden
  behind `#if` guards and inside `@{` blocks are recovered; a brief is
  inferred from the first sentence and then elided from the details body
  rather than duplicated; leaked tags are cleaned up; and orphaned items
  fall back to their group. A C signature renders as a `.. rubric::`
  with a `code-block:: c` instead of a plain backticked line, and C
  `const` is stripped from `:py:obj:` type references.

#### Codegen

- **A fatal clang diagnostic raises.** The danger in a failed include is
  not the missing header; it is that clang recovers, hands back a
  different type, and nothing has failed.
- **The generator runs on Windows.**
- **The 3859-line Cython monolith is a seven-file package**, and the
  codebase was renamed `hip` → `rocm` end to end to match the target
  namespace. Type rendering goes through a single `TypeHandler`, and the
  `.pyi` renderer was refactored onto the node hierarchy.
- **libclang 17+ AST shapes and pyparsing 3.0+ are tolerated**, both
  previously pinned to older versions, and the Cython floor moved to
  `>=3.1.0` for a 3.0.x `*const *` codegen bug. Upstream-bug notes were
  recorded for the hipBLASLt and hipSPARSELt C-includability issues and
  the HIP and amdsmi doxygen pointer-intent mistags; the workarounds
  live in the per-library recipes. A source tree whose commit hash
  cannot be recorded now produces a warning.

### Removed

#### Codegen

- **Commit-count versioning.** `support/gitversion.py` keeps only
  `git_rev()`, `git_current_branch()`, `git_describe()` and
  `git_is_clean()`, and the count-composed `LONG_VERSION` and
  `_CODEGEN_VERSION` are no longer emitted.

### Fixed

#### Bindings

- **Every Python-derived call argument is bound to a local** before the
  C call, in both emitters. The with-GIL emitter inlined the adapter
  into the call expression on the theory that holding the GIL kept the
  temporary alive; it does not, so a malloc'd array was freed before the
  call ran and the callee read reclaimed memory.
- **Type rendering agrees with itself** across typedef-of-typedef
  chains, `EXTVECTOR`, `UNEXPOSED` and unknown `TypeKind` ids, and
  header-tuple extraction edge cases. Foreign-record pointer parameters
  route through `Pointer`, records admit transitively, and anonymous
  function-pointer ctypedefs are emitted transitively.

#### Codegen

- **A `const` pointee survives on the `nogil` retval holder.** The
  emitter stripped every `const`, so a `const char *` return produced an
  assignment Cython warned discarded the qualifier — 14 warnings in the
  LLVM bindings once those modules started releasing the GIL. A `const`
  *value* return still drops it, since Cython rejects the assignment
  into a `cdef const T` local.
- **The builtin-include fallback stands aside for either spelling of
  `-resource-dir`.** Only `-resource-dir DIR` was recognised, so a caller
  writing the equally valid `-resource-dir=DIR` had the fallback appended
  anyway — as `-isystem`, which is searched first, so the three shipped
  headers shadowed the toolchain that caller had just named. Every
  production path uses the separated form, so no binding was generated
  from the wrong `stddef.h`.
