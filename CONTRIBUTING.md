<!-- MIT License
  --
  -- Copyright (c) 2026 Advanced Micro Devices, Inc.
  --
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  --
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  --
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->
# Contributing to HIP Python

This is a routing guide: it tells you *where* a change belongs and *how*
to validate it. The design documents under [`share/design/`](share/design)
carry the detail, and this page links into them rather than repeating
them.

## The one rule

**Never edit the generated bindings in place.** Most of what ships in the
`rocm-bindings-*` and `hip-python-interop` wheels is machine-written:

```text
packages/<wheel>/src/rocm/bindings/<lib>.{pxd,pyx,pyi}     # generated
packages/<wheel>/src/rocm/bindings/cy<lib>.{pxd,pyx}       # generated
packages/hip-python-interop/src/cuda/bindings/*.{pxd,pyx}  # generated
packages/*/cmake/generated_*.cmake                         # generated
```

A fix applied to one of those files survives exactly until the next
release run, when the generator overwrites it. Fix the generator instead
(see below) so the correction is reproduced for every future ROCm
version.

The inverse holds too: the handcoded components listed in the next
section are never touched by the generator, so they are edited directly
in the source tree.

## Which tier is my change in?

| Symptom | Tier |
|---|---|
| A symbol is missing, spurious, or misnamed in a binding | generated |
| A parameter has the wrong direction (IN / OUT / INOUT) | generated |
| A pointer is bound as a scalar instead of a buffer, or vice versa | generated |
| Macros leak into the Python namespace as nonsense constants | generated |
| A generated docstring is wrong or unhelpful | generated |
| A ROCm header fails to parse, or a library needs binding at all | generated |
| Loader, `ROCM_PATH` resolution, shared Cython types (`Pointer`, `CStr`, `ListOf*`, `NDBuffer`) | handcoded |
| `_hip_helpers` / `_hiprtc_helpers`, `rocm.comgr`, `rocm.hipfile`, `rocm.bindings.hiprtc_pyext` | handcoded |
| The `pynvml` / `nvtx` / `cuda.core` compatibility shims, `cuda.bindings.cufile` | handcoded |
| `numba-hip`, the `hip.*` alias package | handcoded |
| CMake, `pyproject.toml`, CI scripts, examples, tests, documentation | handcoded |

The authoritative split is written down in two places and this guide does
not duplicate the file lists:

- [share/design/CODEGEN.md](share/design/CODEGEN.md), sections
  "Generator-owned outputs" and "Forbidden outputs (handcoded; generator
  must NEVER write)".
- [share/design/BINDINGS.md](share/design/BINDINGS.md), section "What is
  NOT generator-controlled".

Branch tiers matter for the same reason: the codegen base branch holds
only handcoded content and is not buildable on its own, while a release
branch is that base branch plus one ROCm version's generator output. See
CODEGEN.md, section "The two branch tiers in `hip-python`".

## Generated tier: which knob to turn

Two files own a library's binding, and they split by question. Ask
whether the C surface was *parsed* wrongly or *assembled* wrongly.

### Parsing and classification — `interfacegen` recipe controls

[`tools/interfacegen/python/interfacegen/support/recipes/rocm.py`](tools/interfacegen/python/interfacegen/support/recipes/rocm.py)
holds one class per library with the hooks that decide what the generator
sees and how it interprets it:

| Hook | Decides |
|---|---|
| `node_filter` | Which declarations are admitted from the header (typically a strict `<lib>*` / `<LIB>_*` prefix admit, plus the useless-macro filters) |
| `ptr_parm_intent` | Whether a pointer parameter is IN, OUT, or INOUT |
| `ptr_rank` | Whether a pointer is a rank-0 scalar or a rank-1 buffer |

Reach for this when the binding compiles but is *wrong*. The `hiptensor`
class is the worked example: its `_MISTAGGED_INOUT` set overrides
upstream doxygen `@param[out]` tags on parameters the caller must
pre-allocate (so they are really INOUT), and its `ptr_rank` override
pins `hiptensorEstimateWorkspaceSize`'s single-element `uint64_t *` OUT
parameter as a scalar rather than a list. Both carry a comment naming the
upstream defect, which is the expected shape for such an override.

The rule chains these hooks plug into are documented in
[share/design/POINTER_ARGUMENTS.md](share/design/POINTER_ARGUMENTS.md).

### Module assembly — `hip-python-generate` generators

`tools/hip-python-generate/src/hip_python_codegen/generators_{hip,libraries,systems,compiler}.py`
holds one `generate_<lib>` function per library that constructs the
`CythonModuleGenerator`: the module name, the `dll` to resolve at
runtime, the status-enum error contract for the lazy loader, extra
`cimport`s, per-module options, and any header workaround.

Reach for this when the change is about how the module is put together
rather than how a declaration is interpreted — a renamed shared library,
a different error return value, an extra `cimport` a generated module
needs. `generate_hiptensor` in
[generators_libraries.py](tools/hip-python-generate/src/hip_python_codegen/generators_libraries.py)
shows all of it in one place, including the workaround that cimports
libc's opaque `FILE` twice so `hiptensorLoggerSetFile` binds.

### Adding, disabling, or removing a library

Adding a library touches four places: a recipe class in `rocm.py`, a
`generate_<lib>` function in the matching `generators_*.py`, an entry in
`AVAILABLE_GENERATORS` in
[binding_generator.py](tools/hip-python-generate/src/hip_python_codegen/binding_generator.py),
and the header-path mapping next to it. POINTER_ARGUMENTS.md section 7,
"How to bind a new ROCm library", walks through it; section 6, "How to
add a new convention", covers the case where the library needs a
classification rule that does not exist yet.

Disabling works the same way in reverse: comment out the
`AVAILABLE_GENERATORS` entry, as done for `hsa`. That single edit removes
the binding from codegen, from the build, and from the documentation at
once.

Removing a binding from the *documentation* only — for a library that is
generated but not compiled into any wheel — is a separate switch:
`_DOCS_EXCLUDED` in
[docs_generator.py](tools/hip-python-generate/src/hip_python_codegen/docs_generator.py)
plus a matching `autoapi_ignore` entry in [docs_src/conf.py](docs_src/conf.py),
since `sphinx-autoapi` reads the committed `.pyi` stubs directly.

### Validating a generated-tier change

1. **Unit tests first.** `pytest tools/interfacegen/python/interfacegen/test`
   is the fast loop and the right place for a regression test. Existing
   tests show the pattern: `test_useless_macros_filtered.py` pins the
   prefix-admit recipes' macro filtering,
   `test_codegen_wrapper_arg_lifetime.py` pins an emitter contract, and
   `test_codegen_gap_libc_file_pointer.py` pins a single upstream
   quirk.
2. **Then regenerate.** For a quick local check, let the build run the
   generator at configure time:

   ```sh
   cd packages
   cmake -B build -DHIP_PYTHON_RUN_CODEGEN=ON \
                  -DHIP_PYTHON_ROCM_PATH=/opt/rocm \
                  -DHIP_PYTHON_ROCM_VERSION=X.Y.Z
   cmake --build build --target all_wheels
   ```

   (Add `-DHIP_PYTHON_FORCE_CODEGEN=ON` to bypass the stamp guard on a
   repeat run.) For the full release-shaped run, use
   [ci/internal/generate-bindings.sh](ci/internal/generate-bindings.sh),
   which additionally needs checkouts of `rocm-systems`,
   `rocm-libraries`, and `llvm-project`, and copies the tree into a
   scratch directory so the generated diff stays isolated. Per-library
   logs land in a temporary directory that the CLI prints up front.
3. **Inspect the diff, not just the result.** The regenerated `.pyx`
   diff is the actual review artifact for a recipe change.

Do not commit regenerated bindings to the codegen base branch. They reach
users through a release branch, authored by
[ci/internal/commit-bindings.sh](ci/internal/commit-bindings.sh).

## Handcoded tier

Edit in place, then build and test:

```sh
ci/internal/build-wheels.sh          # or build-wheels.ps1 on Windows
ci/internal/test.sh                  # or test.ps1
```

`test.sh` runs the suites in one venv against the built wheels: the
`examples/` suite, the interop shim unit tests, the GPU-free
`tests/rocm-bindings-{core,compiler}` suites, and `tests/numba-hip`. All
test suites live outside the importable packages, so they exercise the
*installed* wheels. During development a single suite can be run
directly with `pytest`.

Two things are easy to forget:

- **Stubs.** A handcoded Cython module's `.pyi` is committed to git and
  regenerated by a developer-run CMake target
  (`cmake -B build -DHIP_PYTHON_ENABLE_STUBGEN=ON && cmake --build build
  --target all_stubs`, or [ci/docs/regenerate-stubs.sh](ci/docs/regenerate-stubs.sh)).
  It is *not* produced by the binding generator, and the wheel build does
  not depend on it. Commit `<module>.pyx` and `<module>.pyi` together.
  Never hand-edit a stub carrying the `AUTO-GENERATED` banner — rerun the
  target instead, from a build tree configured without
  `HIP_PYTHON_ABI3_FLOOR`. The one hand-maintained stub is
  `cuda.bindings.cufile`'s, which stubgen cannot produce usefully; the
  `tests/stubs` suite checks that it still covers the module's public
  surface. See [share/design/BUILDING.md](share/design/BUILDING.md),
  section "Regenerating stubs for handcoded Cython modules".
- **Platform guards.** An example or test that needs a library ROCm does
  not ship everywhere must skip with a reason rather than fail. The
  Windows availability matrix is in the user guide
  ([docs_src/user_guide/0_install.rst](docs_src/user_guide/0_install.rst)).

## Documentation

The doc input language is reStructuredText under
[`docs_src/`](docs_src), distinct from the Markdown README and design
documents at the repo root. Build it independently of the wheels:

```sh
cd packages
cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target docs
```

The API reference is generated by `sphinx-autoapi` from the `.pyi`
stubs, so it does not need compiled extensions on `sys.path`. The
table of contents is *not* handwritten: `docs_src/sphinx/_toc.yml.in` is
rendered by the code generator from `_toc.yml.in.in`, so an edit to the
API subtrees belongs in the template or in
[docs_generator.py](tools/hip-python-generate/src/hip_python_codegen/docs_generator.py).
The handwritten user guide pages are listed inline in the template and
can be edited there directly.

## Changelogs and commits

Three changelogs, by what you touched:

| Changelog | Covers |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | The repo at large: bindings, build system, packaging, docs |
| [packages/numba-hip/CHANGELOG.md](packages/numba-hip/CHANGELOG.md) | `numba-hip`, which carries its own version |
| [tools/interfacegen/CHANGELOG.md](tools/interfacegen/CHANGELOG.md) | The generator itself |

Entries go under the topmost version heading (for example `## 0.1.0`),
which is the version in progress until a date is added to it at release.
Inside it, pick an `### Added`, `### Changed`, `### Removed` or
`### Fixed` group, then `#### Bindings`, `#### Codegen` or
`#### Building` — what someone importing `rocm.*` or `cuda.*` sees, the
generator and the artifacts it emits, and the build, the packaging and
the docs pipeline. The generator's own log has no `Building` group. Write
each entry as one bullet opening with the change in bold: what a reader
can now do, what behaves differently, and any step they have to take. A
signature change is worth marking as one. Why the code had to change
belongs in the commit message, and how it works belongs in the design
docs — not here.
Commits follow the conventional-commit style already in use:
`type(scope): summary`, for example `fix(numba-hip): run LLVM passes in a
child that needs no fork`.

## See also

- [share/design/CODEGEN.md](share/design/CODEGEN.md) — the code
  generation pipeline, branch tiers, and the release run.
- [share/design/BINDINGS.md](share/design/BINDINGS.md) — anatomy of a
  generated function: the status-first return tuple, GIL semantics, the
  loader error contract.
- [share/design/POINTER_ARGUMENTS.md](share/design/POINTER_ARGUMENTS.md)
  — pointer intent and rank classification, and how to extend it.
- [share/design/BUILDING.md](share/design/BUILDING.md) — build system
  design, CMake options, stub regeneration.
- [tools/hip-python-generate/README.md](tools/hip-python-generate/README.md)
  — the codegen CLI's flags and wheel catalog.
