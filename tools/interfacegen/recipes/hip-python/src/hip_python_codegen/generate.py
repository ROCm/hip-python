# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unified hip-python code generator.

Produces generator outputs for the five generator-owned packages of
hip-python:

  - rocm-bindings-hip        (hip, hiprtc)
  - rocm-bindings-libraries  (math: hipblas, hipfft, hiprand, hipsolver, hipsparse)
  - rocm-bindings-systems    (collective comm + tracing + I/O: rccl, roctx, hipfile)
  - rocm-bindings-compiler   (amd_comgr, llvm)
  - hip-python-interop       (CUDA interop layer)

Generator-owned outputs:

  * `.pxd` / `.pyx` Cython sources for every high-level Python module
    plus its paired cy*-prefixed C-level wrapper.
  * `.pyi` type-stub files for every **high-level** Python module
    (high-level only — cy* modules are cimport-only and deliberately
    not stubbed; see `share/design/CODEGEN.md`).
  * `__init__.pxd` namespace package markers below `rocm/bindings/`
    and `cuda/bindings/` (build-time only; never installed).
  * `cmake/generated_modules.cmake` (libraries + compiler) — module
    lists consumed by per-package CMakeLists.txt.
  * `cmake/generated_versions.cmake` (every package) — version
    metadata consumed by the `configure_file("_version.py.in" …)`
    flow.
  * `docs_src/python_api/<dotted-name>.rst` Sphinx wrapper pages —
    autoapi pages for high-level modules + literalinclude pages for
    cy* C-level wrappers.

NO Python-packaging files are produced — those (handcoded
`__init__.py`, `_version.py.in`, `pyproject.toml`, `setup.py`, etc.)
live in the hip-python repo as the source of truth. See
`share/design/CODEGEN.md` for the full table.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import binding_generator
from . import docs_generator


# Wheel shortnames accepted by --include / --exclude. Mirrors the keys of
# `binding_generator._PKG_TO_DIR`.
WHEEL_NAMES = ["hip", "libraries", "systems", "compiler"]


# Map recipe name -> generate(opts) callable.
# `comgr` and `llvm` were previously their own recipes; both have been
# merged into the hip recipe as libraries inside
# `binding_generator.AVAILABLE_GENERATORS`. Select wheels via
# `--include` / `--exclude` (default: all four wheels included).
RECIPE_GENERATORS = {
    "hip":   binding_generator.generate,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_unified_args(argv=None):
    p = argparse.ArgumentParser(
        prog="hip-python-generate",
        description=(
            "Unified hip-python code generator. Produces Cython sources for "
            "rocm-bindings-{hip,libraries,systems,compiler} and hip-python-interop."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "output_dir",
        help=(
            "Root of the hip-python repo. The generator writes into "
            "<output_dir>/packages/<package>/..."
        ),
    )
    p.add_argument("--rocm-version", required=True, help="ROCm version (e.g. 7.13.0)")
    p.add_argument("--rocm-path", default=None, help="Path to ROCm installation (optional)")
    p.add_argument("--rocm-systems-dir", default=None, help="Path to rocm-systems repository root")
    p.add_argument("--rocm-libraries-dir", default=None, help="Path to rocm-libraries repository root")
    p.add_argument("--rocm-llvm-project-dir", default=None, help="Path to llvm-project repository root")
    p.add_argument(
        "--clang-resource-dir", default=None,
        help=(
            "Path to libclang's resource directory. If omitted and "
            "--rocm-path is set, derived automatically by running "
            "<rocm-path>/llvm/bin/clang -print-resource-dir."
        ),
    )
    p.add_argument(
        "--recipes",
        default="hip",
        help="Comma-separated subset of {hip}",
    )
    p.add_argument(
        "--include", nargs="+", choices=WHEEL_NAMES, default=list(WHEEL_NAMES),
        metavar="WHEEL",
        help=f"Wheels to generate. Default: all four ({', '.join(WHEEL_NAMES)}).",
    )
    p.add_argument(
        "--exclude", nargs="+", choices=WHEEL_NAMES, default=[],
        metavar="WHEEL",
        help="Wheels to skip. Subtracted from --include. Default: none.",
    )
    p.add_argument(
        "--no-rt-linking", action="store_true",
        help="Disable runtime linking; bind against the named shared libraries directly.",
    )
    p.add_argument("--license-path", default=None)
    p.add_argument(
        "--generator-args", nargs="*", default=[],
        help="Extra args passed through to libclang.",
    )
    args = p.parse_args(argv)
    args.runtime_linking = not args.no_rt_linking
    args.recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in args.recipes:
        if r not in RECIPE_GENERATORS:
            p.error(f"unknown recipe '{r}'; pick from {list(RECIPE_GENERATORS)}")

    # Validation: at least one of rocm-path or repository directories must be specified
    if not any([args.rocm_path, args.rocm_systems_dir, args.rocm_libraries_dir, args.rocm_llvm_project_dir]):
        p.error(
            "At least one header source must be specified: "
            "--rocm-path, --rocm-systems-dir, --rocm-libraries-dir, or --rocm-llvm-project-dir"
        )

    # Auto-detect --clang-resource-dir from --rocm-path. Was previously done
    # by the now-deleted shell wrapper; folding it in here keeps direct
    # invocation ergonomic.
    if not args.clang_resource_dir and args.rocm_path:
        candidate = os.path.join(args.rocm_path, "llvm", "bin", "clang")
        if os.path.isfile(candidate):
            try:
                args.clang_resource_dir = subprocess.check_output(
                    [candidate, "-print-resource-dir"], text=True,
                ).strip()
            except (subprocess.SubprocessError, OSError):
                pass  # leave None; binding_generator raises a clear error if needed

    return args



def main(argv=None):
    opts = parse_unified_args(argv)
    recipe_results = {}
    for recipe in opts.recipes:
        result = RECIPE_GENERATORS[recipe](opts) or {}
        recipe_results[recipe] = result
    binding_generator.write_namespace_markers(opts, recipe_results)
    binding_generator.write_cmake_module_lists(opts, recipe_results)
    binding_generator.write_cmake_version_files(opts, recipe_results)
    binding_generator.write_version_template_file(opts, recipe_results)
    docs_generator.write_docs_pages(opts, recipe_results)
    docs_generator.write_toc_yml_in(opts, recipe_results)

    # Per-library log paths (and any errors) collected from recipes that
    # ran in parallel. Reported at the end so users can grep for failures
    # or inspect verbose libclang output without polluting the main log.
    all_log_paths = {}
    all_errors = {}
    for recipe, result in recipe_results.items():
        for lib, path in (result.get("log_paths") or {}).items():
            all_log_paths[f"{recipe}:{lib}"] = path
        for lib, err in (result.get("errors") or {}).items():
            all_errors[f"{recipe}:{lib}"] = err

    if all_log_paths:
        print("\n=== Code generation log files ===", file=sys.stderr)
        for key in sorted(all_log_paths):
            marker = "  [FAILED]" if key in all_errors else ""
            print(f"  {key}: {all_log_paths[key]}{marker}", file=sys.stderr)

    if all_errors:
        print(
            f"\n[error] {len(all_errors)} library generation(s) failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
