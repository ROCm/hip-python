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

Produces generator outputs for the four generator-owned packages of
hip-python:

  - rocm-bindings-hip
  - rocm-bindings-libraries
  - rocm-bindings-compiler
  - hip-python-interop

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
import sys
from pathlib import Path

from interfacegen.support import gitversion

from .hip import generate_hip
from .llvm import generate_llvm
from .comgr import generate_comgr


# Map recipe name -> generate(opts) callable.
RECIPE_GENERATORS = {
    "hip":   generate_hip.generate,
    "llvm":  generate_llvm.generate,
    "comgr": generate_comgr.generate,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_unified_args(argv=None):
    p = argparse.ArgumentParser(
        prog="hip_python.codegen",
        description=(
            "Unified hip-python code generator. Produces Cython sources for "
            "rocm-bindings-{hip,libraries,compiler} and hip-python-interop."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "output_dir",
        help=(
            "Root of the hip-python repo. The generator writes into "
            "<output_dir>/python/<package>/..."
        ),
    )
    p.add_argument("--rocm-version", required=True, help="ROCm version (e.g. 7.13.0)")
    p.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
    p.add_argument("--platform", default="amd", choices=["amd", "nvidia"])
    p.add_argument("--clang-resource-dir", default=None)
    p.add_argument(
        "--recipes",
        default="hip,llvm,comgr",
        help="Comma-separated subset of {hip,llvm,comgr}",
    )
    p.add_argument(
        "--hip-libs",
        default="*",
        help="HIP library subset, e.g. 'hip,hiprtc,hipblas' or '*' or '^excludes'",
    )
    p.add_argument("--llvm-libs", default="*")
    p.add_argument("--comgr-libs", default="*")
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
    return args


# ---------------------------------------------------------------------------
# Cross-cutting emission: namespace markers + CMake module/version lists
# ---------------------------------------------------------------------------

NAMESPACE_MARKER_BODY = (
    "# Cython namespace package marker — build-time only, not installed.\n"
    "# Auto-generated by the hip-python code generator.\n"
)


def write_namespace_markers(opts, recipe_results):
    """Drop `__init__.pxd` into every generator-created subdirectory.

    Top-of-namespace markers (`rocm/__init__.pxd`, `rocm/bindings/__init__.pxd`,
    `cuda/__init__.pxd`, `cuda/bindings/__init__.pxd`) are HANDCODED in the
    hip-python repo and not touched here (plan §A.1).

    This pass walks each generator-owned subtree (everything strictly below
    `rocm/bindings/` and `cuda/bindings/`) and writes a marker into each
    directory that contains a `.pxd` or `.pyx` but no `__init__.py` /
    `__init__.pxd` yet.
    """
    package_roots = [
        os.path.join(opts.output_dir, "python", "rocm-bindings-hip", "rocm", "bindings"),
        os.path.join(opts.output_dir, "python", "rocm-bindings-libraries", "rocm", "bindings"),
        os.path.join(opts.output_dir, "python", "rocm-bindings-compiler", "rocm", "bindings"),
        os.path.join(opts.output_dir, "python", "hip-python-interop", "cuda", "bindings"),
    ]
    for root in package_roots:
        if not os.path.isdir(root):
            continue
        # First pass: identify every directory that ends up containing a
        # .pxd/.pyx (directly or via a descendant). Then emit __init__.pxd
        # at every such directory below `root` (top-of-namespace is
        # handcoded — skip `root` itself).
        dirs_needing_marker = set()
        for dirpath, _dirnames, filenames in os.walk(root):
            if any(fn.endswith((".pxd", ".pyx")) for fn in filenames):
                cur = dirpath
                while cur != root and cur != os.path.dirname(cur):
                    dirs_needing_marker.add(cur)
                    cur = os.path.dirname(cur)
        for dirpath in dirs_needing_marker:
            existing = os.listdir(dirpath)
            if "__init__.py" in existing or "__init__.pxd" in existing:
                continue
            marker = os.path.join(dirpath, "__init__.pxd")
            with open(marker, "w") as f:
                f.write(NAMESPACE_MARKER_BODY)


def write_cmake_module_lists(opts, recipe_results):
    """Write per-package `cmake/generated_modules.cmake` files.

    Module lists come from each subgenerator's return value:
      - libraries:  comes from the HIP recipe's hip_modules excluding the
                    handcoded core (hip, hiprtc) and helper (_hip_helpers,
                    _hiprtc_helpers) modules. NOTE: today the libraries
                    list is hand-curated; this function preserves the
                    existing list when the HIP recipe didn't run.
      - compiler:   union of llvm_modules and comgr_modules from the LLVM
                    and COMGR recipes.
    """
    # rocm-bindings-libraries
    hip_result = recipe_results.get("hip")
    if hip_result is not None:
        libs = hip_result.get("libraries_modules") or []
        path = os.path.join(
            opts.output_dir, "python", "rocm-bindings-libraries", "cmake",
            "generated_modules.cmake",
        )
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(_AUTOGEN_HEADER)
            f.write(f"set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES\n    {' '.join(libs)})\n")

    # rocm-bindings-compiler
    llvm_modules = recipe_results.get("llvm", {}).get("llvm_modules") or []
    comgr_modules = recipe_results.get("comgr", {}).get("comgr_modules") or []
    if llvm_modules or comgr_modules:
        path = os.path.join(
            opts.output_dir, "python", "rocm-bindings-compiler", "cmake",
            "generated_modules.cmake",
        )
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        # Subdivide LLVM modules by their on-disk subdirectory.
        c_modules, transforms_modules, config_modules = [], [], []
        compiler_root = os.path.join(
            opts.output_dir, "python", "rocm-bindings-compiler", "rocm", "bindings", "llvm",
        )
        for m in llvm_modules:
            # llvm_modules is a list of py_global_name strings like
            # "rocm.bindings.llvm.c.core" — bucket by the subpath.
            parts = m.split(".")
            try:
                idx = parts.index("llvm")
            except ValueError:
                continue
            tail = parts[idx + 1:]  # e.g. ["c", "core"], ["c", "transforms", "passbuilder"]
            if not tail:
                continue
            leaf = tail[-1]
            if "transforms" in tail:
                transforms_modules.append(leaf)
            elif "config" in tail:
                config_modules.append(leaf)
            elif "c" in tail:
                c_modules.append(leaf)
        with open(path, "w") as f:
            f.write(_AUTOGEN_HEADER)
            for var, lst in (
                ("HIP_PYTHON_LLVM_C_MODULES", c_modules),
                ("HIP_PYTHON_LLVM_C_TRANSFORMS_MODULES", transforms_modules),
                ("HIP_PYTHON_LLVM_CONFIG_MODULES", config_modules),
                ("HIP_PYTHON_COMGR_MODULES", [m.split(".")[-1] for m in comgr_modules]),
            ):
                f.write(f"set({var}\n    {' '.join(lst)})\n\n")


def write_cmake_version_files(opts, recipe_results):
    """Write per-package `cmake/generated_versions.cmake` (plan §B.7)."""
    rocm_version = opts.rocm_version
    hip_version = recipe_results.get("hip", {}).get("hip_version")
    if hip_version:
        major, minor, patch, githash = hip_version
        hip_version_str = f"{major}.{minor}.{patch}-{githash}" if githash else f"{major}.{minor}.{patch}"
    else:
        hip_version_str = ""
    try:
        codegen_branch = gitversion.git_current_branch()
        codegen_rev = gitversion.git_rev()
        codegen_version = gitversion.version(append_hash=True, append_date=True)
    except Exception:
        codegen_branch = codegen_rev = codegen_version = ""
    body = (
        _AUTOGEN_HEADER
        + f'set(HIP_PYTHON_GENERATED_ROCM_VERSION    "{rocm_version}")\n'
        + f'set(HIP_PYTHON_GENERATED_HIP_VERSION     "{hip_version_str}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_BRANCH  "{codegen_branch}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_REV     "{codegen_rev}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_VERSION "{codegen_version}")\n'
    )
    for pkg in ("rocm-bindings-hip", "rocm-bindings-libraries",
                "rocm-bindings-compiler", "hip-python-interop"):
        path = os.path.join(opts.output_dir, "python", pkg, "cmake", "generated_versions.cmake")
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(body)


_AUTOGEN_HEADER = (
    "# AUTO-GENERATED by the hip-python code generator. Do not edit by hand.\n"
)

# Header used in generator-emitted .rst pages so the cleanup pass in
# `write_docs_pages` can distinguish them from handcoded pages.
_AUTOGEN_RST_HEADER = (
    ".. AUTO-GENERATED by the hip-python code generator. Do not edit by hand.\n"
)


# ---------------------------------------------------------------------------
# Sphinx .rst page emission per generated module
# ---------------------------------------------------------------------------

def _module_to_pxd_relpath(opts, module_name):
    """Return the .pxd path on disk for a generated module's `cy<name>` wrapper.

    Paths are returned **relative to docs_src/** (so they fit cleanly into a
    `literalinclude` directive). `module_name` is a high-level dotted name
    like `rocm.bindings.hipblas`; the corresponding cy* wrapper lives next
    to it.
    """
    parts = module_name.split(".")
    leaf = parts[-1]
    cy_leaf = leaf if leaf.startswith("cy") else f"cy{leaf}"
    if module_name.startswith("rocm.bindings.llvm."):
        # rocm.bindings.llvm.c.core → python/rocm-bindings-compiler/rocm/bindings/llvm/c/cycore.pxd
        subdir = "/".join(parts[2:-1])  # llvm/c, llvm/c/transforms, llvm/config
        rel = f"python/rocm-bindings-compiler/rocm/bindings/{subdir}/{cy_leaf}.pxd"
    elif module_name.startswith("rocm.bindings."):
        # decide hip vs libraries vs compiler by leaf name
        _HIP_CORE = {"hip", "hiprtc", "_hip_helpers", "_hiprtc_helpers"}
        _COMPILER_CORE = {"amd_comgr"}
        bare = leaf[2:] if leaf.startswith("cy") else leaf
        if bare in _HIP_CORE:
            pkg = "rocm-bindings-hip"
        elif bare in _COMPILER_CORE:
            pkg = "rocm-bindings-compiler"
        else:
            pkg = "rocm-bindings-libraries"
        rel = f"python/{pkg}/rocm/bindings/{cy_leaf}.pxd"
    elif module_name.startswith("cuda.bindings."):
        rel = f"python/hip-python-interop/cuda/bindings/{cy_leaf}.pxd"
    else:
        return None
    # docs_src/python_api/<page>.rst → ../../<rel>
    return f"../../{rel}"


def _all_emitted_modules(recipe_results):
    """Collect every dotted module name produced across all subgenerators."""
    modules = []

    def _add(name):
        if name and name not in modules:
            modules.append(name)

    hip_result = recipe_results.get("hip") or {}
    for name in hip_result.get("hip_modules") or []:
        _add(f"rocm.bindings.{name}")
    # The HIP recipe also emits CUDA interop modules.
    for name in ("driver", "runtime", "nvrtc"):
        _add(f"cuda.bindings.{name}")

    llvm_result = recipe_results.get("llvm") or {}
    for global_name in llvm_result.get("llvm_modules") or []:
        # global_name is already in dotted form (rocm.bindings.llvm.c.core).
        _add(global_name)

    comgr_result = recipe_results.get("comgr") or {}
    for global_name in comgr_result.get("comgr_modules") or []:
        _add(global_name)

    return modules


def write_docs_pages(opts, recipe_results):
    """Emit one Sphinx .rst page per generated module.

    For each high-level module, emit an autoapi-driven page. For each
    cy* C-level wrapper, emit a page that uses `literalinclude` to embed
    the .pxd source with Cython syntax highlighting (the .pxd itself is
    the readable contract for downstream Cython users).

    Cleanup: any existing `.rst` (or leftover `.md`) under
    docs_src/python_api/ that starts with the autogen header AND no longer
    corresponds to a current module is removed. Handcoded pages
    (`rocm.bindings.util.rst`, `hip.rst`, …) lack the header and are
    preserved.
    """
    docs_dir = os.path.join(opts.output_dir, "docs_src", "python_api")
    Path(docs_dir).mkdir(parents=True, exist_ok=True)

    high_level = _all_emitted_modules(recipe_results)
    expected = set()
    for module in high_level:
        expected.add(f"{module}.rst")
        # Each high-level module has a paired cy* page (except for
        # _hip_helpers / _hiprtc_helpers which are handcoded and therefore
        # not in the high-level list).
        leaf = module.rsplit(".", 1)[-1]
        if not leaf.startswith("cy") and not leaf.startswith("_"):
            cy_dotted = ".".join(module.rsplit(".", 1)[:-1] + [f"cy{leaf}"])
            expected.add(f"{cy_dotted}.rst")

    # Emit pages.
    for module in high_level:
        leaf = module.rsplit(".", 1)[-1]
        if leaf.startswith("_"):
            continue  # handcoded helpers (_hip_helpers, _hiprtc_helpers)
        # High-level Python autoapi page.
        underline = "=" * len(module)
        with open(os.path.join(docs_dir, f"{module}.rst"), "w") as f:
            f.write(
                f"{_AUTOGEN_RST_HEADER}\n"
                f"{module}\n{underline}\n\n"
                f".. autoapi-module:: {module}\n"
            )
        # cy* C-level literalinclude page (skip if this entry already IS cy*).
        if leaf.startswith("cy"):
            continue
        cy_dotted = ".".join(module.rsplit(".", 1)[:-1] + [f"cy{leaf}"])
        cy_underline = "=" * len(cy_dotted)
        pxd_rel = _module_to_pxd_relpath(opts, module)
        if pxd_rel is None:
            continue
        with open(os.path.join(docs_dir, f"{cy_dotted}.rst"), "w") as f:
            f.write(
                f"{_AUTOGEN_RST_HEADER}\n"
                f"{cy_dotted}\n{cy_underline}\n\n"
                f"Cython declarations for the C-level interface. Use\n"
                f"``cimport {cy_dotted}`` from your Cython code to access\n"
                f"these declarations directly.\n\n"
                f".. seealso::\n\n"
                f"   The high-level Python API is documented at "
                f":doc:`{module}`.\n\n"
                f".. literalinclude:: {pxd_rel}\n"
                f"   :language: cython\n"
            )

    # Cleanup pass: remove autogen pages whose module is no longer present.
    expected_str = {n for n in expected if n}
    for fn in os.listdir(docs_dir):
        if fn in expected_str:
            continue
        if not (fn.endswith(".rst") or fn.endswith(".md")):
            continue
        path = os.path.join(docs_dir, fn)
        try:
            with open(path) as f:
                head = f.read(len(_AUTOGEN_RST_HEADER) * 2)
        except OSError:
            continue
        if _AUTOGEN_RST_HEADER.strip() in head or "<!-- AUTO-GENERATED" in head:
            try:
                os.remove(path)
            except OSError:
                pass


def write_toc_yml_in(opts, recipe_results):
    """Render `_toc.yml.in.in` → `_toc.yml.in` with discovered modules.

    The handcoded template at `<output_dir>/docs_src/sphinx/_toc.yml.in.in`
    contains a `@TOC_ENTRIES_<SECTION>@` placeholder per generator-owned
    subtree (rocm-bindings-{hip,libraries,compiler}, hip-python-interop,
    cython-level). This pass substitutes each placeholder with the
    discovered `      - file: python_api/<dotted-name>` entries.

    Handcoded subtrees (User Guide, rocm-bindings-util, hip compat shim,
    Manual API) are listed inline in the template and left unchanged.
    """
    template_path = os.path.join(
        opts.output_dir, "docs_src", "sphinx", "_toc.yml.in.in"
    )
    output_path = os.path.join(
        opts.output_dir, "docs_src", "sphinx", "_toc.yml.in"
    )
    if not os.path.exists(template_path):
        return  # template missing; skip silently (e.g. truncated checkout)

    # Bucket every emitted module into its TOC section.
    sections = {
        "ROCM_BINDINGS_HIP": [],
        "ROCM_BINDINGS_LIBRARIES": [],
        "ROCM_BINDINGS_COMPILER": [],
        "HIP_PYTHON_INTEROP": [],
        "CYTHON_LEVEL": [],
    }
    _HIP_CORE = {"hip", "hiprtc"}
    _COMPILER_CORE = {"amd_comgr"}
    for module in _all_emitted_modules(recipe_results):
        leaf = module.rsplit(".", 1)[-1]
        if leaf.startswith("_"):
            continue  # handcoded helpers
        if module.startswith("rocm.bindings.llvm."):
            section = "ROCM_BINDINGS_COMPILER"
        elif module.startswith("rocm.bindings."):
            bare = leaf[2:] if leaf.startswith("cy") else leaf
            if bare in _HIP_CORE:
                section = "ROCM_BINDINGS_HIP"
            elif bare in _COMPILER_CORE:
                section = "ROCM_BINDINGS_COMPILER"
            else:
                section = "ROCM_BINDINGS_LIBRARIES"
        elif module.startswith("cuda.bindings."):
            section = "HIP_PYTHON_INTEROP"
        else:
            continue
        sections[section].append(f"      - file: python_api/{module}")
        # Each high-level entry has a paired cy* entry (in the cy subtree).
        if not leaf.startswith("cy"):
            cy_dotted = ".".join(module.rsplit(".", 1)[:-1] + [f"cy{leaf}"])
            sections["CYTHON_LEVEL"].append(
                f"      - file: python_api/{cy_dotted}"
            )

    with open(template_path) as f:
        rendered = f.read()
    for var, entries in sections.items():
        rendered = rendered.replace(
            f"@TOC_ENTRIES_{var}@", "\n".join(entries)
        )

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(_AUTOGEN_TOC_HEADER)
        f.write(rendered)


_AUTOGEN_TOC_HEADER = (
    "# AUTO-GENERATED by the hip-python code generator from\n"
    "# `_toc.yml.in.in`. Do not edit by hand.\n"
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    opts = parse_unified_args(argv)
    recipe_results = {}
    for recipe in opts.recipes:
        result = RECIPE_GENERATORS[recipe](opts) or {}
        recipe_results[recipe] = result
    write_namespace_markers(opts, recipe_results)
    write_cmake_module_lists(opts, recipe_results)
    write_cmake_version_files(opts, recipe_results)
    write_docs_pages(opts, recipe_results)
    write_toc_yml_in(opts, recipe_results)


if __name__ == "__main__":
    sys.exit(main())
