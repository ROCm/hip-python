# hip-python-codegen

Code generator for the [hip-python](https://github.com/ROCm/hip-python) ROCm
Python bindings. Parses the ROCm C headers (and optionally the
`rocm-systems` / `rocm-libraries` / `llvm-project` source repositories) and
emits Cython `.pxd`/`.pyx` files plus the cmake module/version include
files into a hip-python checkout.

After the comgr and llvm recipes were merged into the unified hip recipe,
this is the single tool that produces every generator-owned output for
the four `rocm-bindings-*` wheels and the `hip-python-interop` CUDA shim.

## Install

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r dev-requirements.txt   # installs interfacegen (path-relative)
pip install .                         # installs this codegen + runtime deps
```

The CLI command `hip-python-generate` lands on your PATH.

## Usage

```sh
hip-python-generate <output_dir> --rocm-version X.Y.Z [--rocm-path /opt/rocm]
```

`<output_dir>` is the root of a hip-python checkout. The generator writes
into `<output_dir>/packages/<package>/...`.

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--rocm-version X.Y.Z` | required | The ROCm version being generated for. |
| `--rocm-path PATH` | none | ROCm installation root. If set, `<rocm-path>/llvm/bin/clang -print-resource-dir` is consulted automatically when `--clang-resource-dir` is not provided. |
| `--rocm-systems-dir PATH` | none | Path to a checked-out `rocm-systems` repository. Used as a fallback header source when `--rocm-path` doesn't have the file (also enables in-memory `.h.in` template rendering for RCCL). |
| `--rocm-libraries-dir PATH` | none | Same idea for the `rocm-libraries` repository. |
| `--rocm-llvm-project-dir PATH` | none | Same idea for `llvm-project` (enables in-memory `amd_comgr.h.in` rendering). |
| `--clang-resource-dir PATH` | auto from `--rocm-path` | libclang resource directory. Override only for non-standard clang installs. |
| `--include WHEEL [WHEEL ...]` | all four | Wheels to generate: any of `hip`, `libraries`, `systems`, `compiler`. |
| `--exclude WHEEL [WHEEL ...]` | none | Wheels to skip. Subtracted from `--include`. |
| `--no-rt-linking` | off | Bind directly against named shared libraries instead of resolving at runtime. |
| `--license-path PATH` | none | License header to embed in generated files. |
| `--generator-args ARGS ...` | empty | Extra args passed through to libclang. |

At least one of `--rocm-path`, `--rocm-systems-dir`, `--rocm-libraries-dir`,
or `--rocm-llvm-project-dir` must be provided.

### Wheel catalog

| Wheel (use with `--include` / `--exclude`) | Libraries it covers |
|---|---|
| `hip` | `hip`, `hiprtc` |
| `systems` | `rccl`, `roctx`, `hipfile` |
| `libraries` | `hipblas`, `hiprand`, `hipfft`, `hipsparse`, `hipsolver` |
| `compiler` | `amd_comgr`, `llvm` (multi-module — every llvm-c/* header is emitted) |

### Examples

Generate everything against an installed ROCm:

```sh
hip-python-generate /path/to/hip-python --rocm-version 7.2.0 --rocm-path /opt/rocm
```

Generate only the `hip` and `libraries` wheels:

```sh
hip-python-generate /path/to/hip-python \
    --rocm-version 7.2.0 --rocm-path /opt/rocm \
    --include hip libraries
```

Skip the `compiler` wheel:

```sh
hip-python-generate /path/to/hip-python \
    --rocm-version 7.2.0 --rocm-path /opt/rocm \
    --exclude compiler
```

Only the `compiler` wheel (amd_comgr + every LLVM-C module):

```sh
hip-python-generate /path/to/hip-python \
    --rocm-version 7.2.0 --rocm-path /opt/rocm \
    --include compiler
```

Generate against a checkout of the `rocm-systems` and `llvm-project`
repositories — useful when working on header changes that aren't yet in
an installed ROCm release:

```sh
hip-python-generate /path/to/hip-python \
    --rocm-version 7.2.0 \
    --rocm-systems-dir   /src/rocm-systems \
    --rocm-llvm-project-dir /src/llvm-project \
    --clang-resource-dir $(/opt/rocm/llvm/bin/clang -print-resource-dir)
```

The generator falls back to repository headers when ROCM_PATH doesn't
have a needed header, and renders `.h.in` templates (RCCL's
`nccl.h.in`, COMGR's `amd_comgr.h.in`) in-memory using version metadata
from the repos.

## Logs

Each library gets its own log under
`/tmp/hip_python_codegen_<pid>_<random>/<lib>.log`. The CLI prints the
directory path at the start and a summary table at the end. Failures
surface there alongside any libclang diagnostic spam, keeping the
top-level output readable.

## What this tool produces

For each run, it writes:

- `.pxd` / `.pyx` Cython sources for every high-level Python module
  plus its paired `cy*`-prefixed C-level wrapper.
- `.pyi` type-stub files for every high-level module.
- `__init__.pxd` namespace package markers under `rocm/bindings/` and
  `cuda/bindings/`.
- `cmake/generated_modules.cmake` per affected wheel.
- `cmake/generated_versions.cmake` per affected wheel.
- `docs_src/python_api/<dotted-module-name>.rst` Sphinx wrapper pages.

It does NOT produce hand-coded files (`__init__.py`, `_version.py.in`,
`pyproject.toml`, `setup.py`, etc.). Those live in the hip-python repo
on the codegen base branch as the source of truth.

## Repository layout

```
recipes/hip-python/
├── pyproject.toml             # CLI entry-point + runtime deps
├── README.md                  # this file
├── dev-requirements.txt       # interfacegen path install
└── src/hip_python_codegen/
    ├── generate.py            # CLI: argparse + main() dispatcher
    ├── binding_generator.py   # master orchestrator + cmake/marker writers
    ├── docs_generator.py      # Sphinx page + TOC YAML emission
    ├── generators_hip.py      # hip + hiprtc generators
    ├── generators_libraries.py # math libs (hipblas/hipfft/...)
    ├── generators_systems.py  # rccl + roctx + hipfile generators
    ├── generators_compiler.py # amd_comgr + llvm generators
    ├── cuda_interop.py        # CUDA interop subgenerator
    └── hipify.py              # hipify-perl substitution parser
```
