# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import datetime
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import cuda_interop as cuda_interop_layer_gen
from . import generators_compiler
from . import generators_hip
from . import generators_libraries
from . import generators_systems
from .hipify import parse_hipify_perl

import interfacegen
from interfacegen import template_renderer
from interfacegen.cython import CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER

interfacegen.enable_logging(logging.INFO)
_log = logging.getLogger("interfacegen")

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = (
    r"`~.{name}`"  # ~: removes the qualifier from the link text
)
cuda_interop_layer_gen.python_interface_pyobj_role_template = (
    r"`.{name}`"  # note: here we want to keep the qualifier
)

HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER = (
    CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER("rocm.bindings.util.types.")
)


# ==============================================================================
# Header Resolution and Template Rendering
# ==============================================================================


def get_systems_header(header_relpath: str, rocm_systems_dir: str):
    """Get header from rocm-systems repository, handling .h.in templates.

    Returns:
        Tuple of (header_path, rendered_content) or None if not found.
        - header_path: Absolute path to header file (or template)
        - rendered_content: String content if template was rendered, None if using file directly
    """
    # Map header paths to repository locations
    mappings = {
        "hipfile.h": "projects/hipfile/include/hipfile.h",
        "roctracer/roctx.h": "projects/roctracer/inc/roctracer/roctx.h",
        "rccl/rccl.h": "projects/rccl/src/rccl.h",  # or nccl.h.in
        "hip/hip_runtime_api.h": "projects/hip/include/hip/hip_runtime_api.h",
        "hip/hiprtc.h": "projects/hip/include/hip/hiprtc.h",
    }

    if header_relpath not in mappings:
        return None

    repo_path = mappings[header_relpath]
    full_path = os.path.join(rocm_systems_dir, repo_path)

    # Check for pre-rendered .h file first
    if os.path.exists(full_path):
        return (full_path, None)

    # Special handling for RCCL template
    if header_relpath == "rccl/rccl.h":
        template_path = os.path.join(rocm_systems_dir, "projects/rccl/src/nccl.h.in")
        if os.path.exists(template_path):
            # Parse version and render
            version_mk = os.path.join(rocm_systems_dir, "projects/rccl/makefiles/version.mk")
            try:
                variables = template_renderer.parse_rccl_version(version_mk)
                content = template_renderer.render_template(template_path, variables)
                # Return a *virtual* abspath that matches the install layout
                # (`rccl/rccl.h`). _resolve_include_dir() strips
                # `rccl/rccl.h` from this to derive the include dir;
                # libclang receives the same path via unsaved_files so the
                # in-memory content is served when the file is parsed.
                virtual_abspath = os.path.join(
                    rocm_systems_dir, "projects/rccl/src/rccl/rccl.h",
                )
                return (virtual_abspath, content)
            except (FileNotFoundError, ValueError) as e:
                _log.warning(f"Failed to render RCCL template: {e}")
                return None

    return None


def get_libraries_header(header_relpath: str, rocm_libraries_dir: str):
    """Get header from rocm-libraries repository.

    Returns:
        Tuple of (header_path, rendered_content) or None if not found.
    """
    # Map header paths to repository locations
    mappings = {
        "hipblas/hipblas.h": "projects/hipblas/library/include/hipblas/hipblas.h",
        "hipblaslt/hipblaslt.h": "projects/hipblaslt/library/include/hipblaslt/hipblaslt.h",
        "hipsolver/hipsolver.h": "projects/hipsolver/library/include/hipsolver/hipsolver.h",
        "hiprand/hiprand.h": "projects/hiprand/library/include/hiprand/hiprand.h",
        "hipsparse/hipsparse.h": "projects/hipsparse/library/include/hipsparse/hipsparse.h",
        "hipfft/hipfft.h": "projects/hipfft/library/include/hipfft/hipfft.h",
        "hipfft/hipfftXt.h": "projects/hipfft/library/include/hipfft/hipfftXt.h",
        "hiptensor/hiptensor.h": "projects/hiptensor/library/include/hiptensor/hiptensor.h",
        "hipsparselt/hipsparselt.h": "projects/hipsparselt/library/include/hipsparselt/hipsparselt.h",
        "hipdnn_backend.h": "projects/hipdnn/backend/include/hipdnn_backend.h",
    }

    if header_relpath not in mappings:
        return None

    repo_path = mappings[header_relpath]
    full_path = os.path.join(rocm_libraries_dir, repo_path)

    # Check if file exists
    if os.path.exists(full_path):
        return (full_path, None)

    return None


def get_llvm_header(header_relpath: str, rocm_llvm_project_dir: str):
    """Get header from llvm-project repository, handling amd_comgr.h.in template.

    Returns:
        Tuple of (header_path, rendered_content) or None if not found.
    """
    # Keys match the install-layout relpath used in `AVAILABLE_GENERATORS`
    # (`amd_comgr/amd_comgr.h`) so the Cython `extern from` line is consistent
    # with how downstream consumers `#include <amd_comgr/amd_comgr.h>`.
    mappings = {
        "amd_comgr/amd_comgr.h": "amd/comgr/include/amd_comgr.h",
    }

    if header_relpath not in mappings:
        return None

    repo_path = mappings[header_relpath]
    full_path = os.path.join(rocm_llvm_project_dir, repo_path)

    # Check for pre-rendered .h file first
    if os.path.exists(full_path):
        return (full_path, None)

    # Handle amd_comgr.h.in template
    if header_relpath == "amd_comgr/amd_comgr.h":
        template_path = os.path.join(rocm_llvm_project_dir, "amd/comgr/include/amd_comgr.h.in")
        if os.path.exists(template_path):
            version_txt = os.path.join(rocm_llvm_project_dir, "amd/comgr/VERSION.txt")
            try:
                variables = template_renderer.parse_comgr_version(version_txt)
                # Add additional CMake variables that might be in template
                variables.update({
                    "AMD_COMGR_EXPORT_DECORATOR": "",  # Empty for Python bindings
                    "AMD_COMGR_DEPRECATED": "",
                })
                content = template_renderer.render_template(template_path, variables)
                # Return a *virtual* abspath that matches the install layout
                # (`amd_comgr/amd_comgr.h`). _resolve_include_dir() strips
                # `amd_comgr/amd_comgr.h` from this to derive the include dir;
                # libclang receives the same path via unsaved_files so the
                # in-memory content is served when the file is parsed.
                virtual_abspath = os.path.join(
                    rocm_llvm_project_dir,
                    "amd/comgr/include/amd_comgr/amd_comgr.h",
                )
                return (virtual_abspath, content)
            except (FileNotFoundError, ValueError) as e:
                _log.warning(f"Failed to render COMGR template: {e}")
                return None

    return None


def _apply_header_workarounds(header_relpath: str, header_path: str, content: str | None):
    """Apply per-header source patches before libclang sees the file.

    Returns (path, content). If `content` is None on entry, reads
    `header_path` from disk only when a workaround applies for this
    header_relpath; otherwise leaves content as None (caller will use
    the on-disk file directly).

    Currently patches:

    - `hipblaslt/hipblaslt.h` — strips the three unconditional C++
      stdlib `#include` lines (`<memory>`, `<regex>`, `<vector>`)
      that prevent the otherwise-C-compatible header from parsing
      under libclang's `-x c` mode. Those includes are unused
      anywhere in the public C API; the C++ extension API lives in
      sibling `hipblaslt-ext.hpp`.
      TODO: remove this workaround once the upstream bug is fixed
      (track ROCm/hipBLASLt issue once filed).
    """
    if header_relpath == "hipblaslt/hipblaslt.h":
        if content is None:
            with open(header_path) as f:
                content = f.read()
        for bad in ("#include <memory>", "#include <regex>", "#include <vector>"):
            content = content.replace(bad, f"// {bad}  /* stripped by hip-python codegen: see _apply_header_workarounds */")
    return (header_path, content)


def resolve_header_path(
    header_relpath: str,
    rocm_inc: str | None,
    rocm_systems_dir: str | None,
    rocm_libraries_dir: str | None,
    rocm_llvm_project_dir: str | None
):
    """Resolve header file path from candidate locations with precedence.

    Handles in-memory rendering of .h.in templates from repositories
    and per-header source workarounds (see `_apply_header_workarounds`).

    Args:
        header_relpath: Relative path like "hipfile.h", "roctracer/roctx.h", "rccl/rccl.h"
        rocm_inc: ${ROCM_PATH}/include (may be None)
        rocm_systems_dir: Path to rocm-systems repo root
        rocm_libraries_dir: Path to rocm-libraries repo root
        rocm_llvm_project_dir: Path to llvm-project repo root

    Returns:
        Tuple of (header_path, rendered_content):
        - header_path: Absolute path to header file (or template)
        - rendered_content: String content if template was rendered or
          patched in-memory, None if using file directly

    Raises:
        FileNotFoundError: If header not found in any candidate location
    """
    located = None  # (path, content) once we find it
    # Priority 1: ROCM_INC (allows patching existing installation)
    if rocm_inc:
        candidate = os.path.join(rocm_inc, header_relpath)
        if os.path.exists(candidate):
            located = (candidate, None)  # Use file directly, no rendering

    # Priority 2: Repository-specific paths
    if located is None and rocm_systems_dir:
        result = get_systems_header(header_relpath, rocm_systems_dir)
        if result:
            located = result

    if located is None and rocm_libraries_dir:
        result = get_libraries_header(header_relpath, rocm_libraries_dir)
        if result:
            located = result

    if located is None and rocm_llvm_project_dir:
        result = get_llvm_header(header_relpath, rocm_llvm_project_dir)
        if result:
            located = result

    if located is not None:
        return _apply_header_workarounds(header_relpath, located[0], located[1])

    # Build error message with all checked locations
    candidates = []
    if rocm_inc:
        candidates.append(os.path.join(rocm_inc, header_relpath))
    if rocm_systems_dir:
        candidates.append(f"{rocm_systems_dir}/projects/*/{{include,inc,src}}/{header_relpath}")
    if rocm_libraries_dir:
        candidates.append(f"{rocm_libraries_dir}/projects/*/library/include/{header_relpath}")
    if rocm_llvm_project_dir:
        candidates.append(f"{rocm_llvm_project_dir}/amd/*/include/{header_relpath}")

    raise FileNotFoundError(
        f"Header '{header_relpath}' not found in any candidate location:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )


def build_generator_include_paths(
    rocm_inc: str | None,
    rocm_systems_dir: str | None,
    rocm_libraries_dir: str | None,
    rocm_llvm_project_dir: str | None,
) -> list[str]:
    """Build -I include paths for clang libclang parser.

    Precedence (highest to lowest):
    1. ROCM_PATH/include (if specified)
    2. Repository include directories

    Always defines __HIP_PLATFORM_AMD__; the codegen targets AMD.

    Returns:
        List like: ["-D", "__HIP_PLATFORM_AMD__", "-I", "/opt/rocm/include", ...]
    """
    cflags = ["-D", "__HIP_PLATFORM_AMD__"]

    # Priority 1: ROCM_PATH/include (highest precedence for "patching")
    if rocm_inc:
        cflags.extend(["-I", rocm_inc])

    # Priority 2: Repository include directories
    if rocm_systems_dir:
        # Add all relevant project include dirs
        for proj_inc in [
            "projects/hip/include",
            "projects/hipfile/include",
            "projects/roctracer/inc",
            "projects/rccl/src",  # Where rendered rccl.h lives
            "projects/amdsmi/include",
        ]:
            path = os.path.join(rocm_systems_dir, proj_inc)
            if os.path.exists(path):
                cflags.extend(["-I", path])

    if rocm_libraries_dir:
        for proj_inc in [
            "projects/hipblas/library/include",
            "projects/hipblaslt/library/include",
            "projects/hipsolver/library/include",
            "projects/hiprand/library/include",
            "projects/hipsparse/library/include",
            "projects/hipfft/library/include",
            "projects/hiptensor/library/include",
            "projects/hipsparselt/library/include",
            "projects/hipdnn/backend/include",
        ]:
            path = os.path.join(rocm_libraries_dir, proj_inc)
            if os.path.exists(path):
                cflags.extend(["-I", path])

    if rocm_llvm_project_dir:
        for proj_inc in [
            "amd/comgr/include",
            "llvm/include",
        ]:
            path = os.path.join(rocm_llvm_project_dir, proj_inc)
            if os.path.exists(path):
                cflags.extend(["-I", path])

    return cflags


# ==============================================================================
# Parallel Code Generation Worker
# ==============================================================================


def _resolve_include_dir(header_path: str, header_relpath: str) -> str:
    """Strip header_relpath from header_path to get the include base dir."""
    rel_components = header_relpath.count(os.sep) + 1
    include_dir = header_path
    for _ in range(rel_components):
        include_dir = os.path.dirname(include_dir)
    return include_dir


def _worker_generate_library(
    libname: str,
    output_dir_root: str,
    rocm_inc,
    rocm_systems_dir,
    rocm_libraries_dir,
    rocm_llvm_project_dir,
    runtime_linking: bool,
    generator_args: list,
    rocm_version_tuple: tuple,
    log_path: str,
):
    """Worker that runs in a subprocess to generate one library's bindings.

    Returns:
        (libname, log_path, elapsed_seconds, status, module_names, error_msg)
        - status: "ok" or "error".
        - module_names: list[str] of dotted names for multi-module libraries
          (currently: llvm); None for single-module libraries. Surfaced so
          the orchestrator can populate `llvm_modules` in its result dict
          for cmake/generated_modules.cmake bucketing.
    """
    # Reconfigure root logger so all output for this library lands in its
    # own file. The interfacegen.enable_logging() call at module import
    # has already attached a stderr handler — replace it.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.INFO)

    # Per-process default ptr handler — closures aren't easily picklable so
    # we rebuild it inside the worker.
    default_ptr_handler = HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER

    start = time.time()
    try:
        callable_, pkg_short, header_relpath = AVAILABLE_GENERATORS[libname]
        pkg_dir = os.path.join(output_dir_root, "packages", *_PKG_TO_DIR[pkg_short])
        Path(pkg_dir).mkdir(parents=True, exist_ok=True)

        if header_relpath is None:
            # Multi-module library (currently: llvm). The callable owns
            # its own tree walk AND its own writing; we hand it
            # output_dir + the LLVM include base.
            if not rocm_inc:
                raise RuntimeError(
                    f"'{libname}' requires --rocm-path to derive llvm/include"
                )
            include_dir = os.path.join(os.path.dirname(rocm_inc), "llvm", "include")
            kwargs = dict(
                output_dir=pkg_dir,
                include_dir=include_dir,
                runtime_linking=runtime_linking,
                generator_args=generator_args,
                default_ptr_handler=default_ptr_handler,
            )
            module_names = callable_(**kwargs)  # returns list[str]
        else:
            # Single-module library: standard two-step.
            header_path, header_content = resolve_header_path(
                header_relpath,
                rocm_inc,
                rocm_systems_dir,
                rocm_libraries_dir,
                rocm_llvm_project_dir,
            )
            include_dir = _resolve_include_dir(header_path, header_relpath)

            kwargs = dict(
                include_dir=include_dir,
                header_relpath=header_relpath,
                runtime_linking=runtime_linking,
                generator_args=generator_args,
                default_ptr_handler=default_ptr_handler,
            )
            if header_content is not None:
                kwargs["header_content"] = header_content
            if libname == "hiprtc":
                kwargs["rocm_version_tuple"] = rocm_version_tuple

            result = callable_(**kwargs)
            # Only the 'hip' generator returns a tuple; hip is never in
            # the parallel branch, so result here is always a single generator.
            generator = result
            generator.write_module_files(output_dir=pkg_dir)
            module_names = None

        elapsed = time.time() - start
        fh.close()
        return (libname, log_path, elapsed, "ok", module_names, None)
    except Exception as e:
        logging.exception(f"Failed to generate {libname}")
        elapsed = time.time() - start
        fh.close()
        return (libname, log_path, elapsed, "error", None, str(e))




# NOTE: helpers that previously wrote `_version.py.in`, `__init__.py`,
# `requirements.txt.in`, and Sphinx docs files were removed during the
# generator-output scope cleanup (plan §B.7). The hip-python repo owns
# those files as handcoded sources; the generator only emits Cython
# bindings and CMake module-list includes.


def generate_cuda_interop_layer_files(
    *,
    license_text: str,
    output_dir: str,
    hip_generator,
    hiprtc_generator,
    hip_2_cuda: dict,
    rocm_version_tuple: tuple,
):
    """Generate the CUDA interoperability layer.

    All state is passed as kwargs from ``generate(opts)`` — no module-level
    globals.

    Note:
        Some CUDA Driver and Runtime routines, namely cuLink*, have been mapped to HIPRTC instead of the HIP runtime.

    Note:
        CUDA Python's `cudaRuntimeGetVersion(...)` returns the version of the
        CUDA version that has been used to generate the bindings. This might
        differ (at least in the patch version) from the version of the CUDA
        runtime that a user might use the bindings for; more details:
        <https://github.com/NVIDIA/cuda-python/issues/16>
        HIP Python's `hipRuntimeGetVersion` has always been calling into the
        loaded runtime so there is no need for a `getLocalRuntimeVersion`.
        In the CUDA compatibility layer's modules, we make
        `getLocalRuntimeVersion` an alias of `hipRuntimeGetVersion`.
    """
    if hiprtc_generator is None or hip_generator is None:
        _log.warning(
            "No CUDA runtime layer generated as 'hip' and/or 'hiprtc' have not been specified as libraries to parse."
        )
        return

    # See: https://github.com/NVIDIA/cuda-python/issues/16
    assert "hipRuntimeGetVersion" in hip_2_cuda
    hip_2_cuda["hipRuntimeGetVersion"].append("getLocalRuntimeVersion")

    if rocm_version_tuple[:2] >= (6, 4):
        # NOTE: Hipify may lag behind the header files.
        #       So we remove outdated keys and ensure
        #       the values are equal to the new names.
        for bad_name in ("hiprtcJITInputType", "hiprtcJIT_option"):
            try:
                del hip_2_cuda[bad_name]
            except KeyError:
                pass
        hip_2_cuda["hipJitInputType"] = [
            "CUjitInputType",
            "CUjitInputType_enum",
        ]
        hip_2_cuda["hipJitOption"] = ["CUjit_option", "CUjit_option_enum"]

    def collect_imports_(import_stmt: str, py_generator):
        contribs = ""
        for node in py_generator:
            hip_name = node.cython_global_name
            if hip_name in hip_2_cuda:
                for cuda_name in hip_2_cuda[hip_name]:
                    contribs += f"{import_stmt} {cuda_name}\n"
        return contribs

    # Modern layout: only emit cuda.bindings.{driver,runtime,nvrtc} with the
    # `cy` prefix on C-level wrappers. The legacy cuda.{cuda,cudart} flat
    # modules are no longer produced (plan §B.6 / §B.5).
    for config in (
        dict(
            parent_package="cuda.bindings",
            driver_name="driver",
            runtime_name="runtime",
            cmodule_prefix="cy",
        ),
    ):
        # write nvrtcm module files
        parent_package = config["parent_package"]
        cmodule_prefix = config["cmodule_prefix"]
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            output_dir,
            f"{parent_package}.nvrtc",
            hiprtc_generator,
            hip_2_cuda,
            license_text,
            cuda_cmodule_prefix=cmodule_prefix,
        )

        # NOTE: hiprtc functions such as hiprtcLink* correspond to cuLink*
        #       functions associated with the driver/runtime and not with nvrtc.
        #       With the below trick we ensure that cuLink* symbols are present
        #       in both driver/runtime and the nvrtc interop layer.
        cuda_extra_args = dict(
            extra_imports=collect_imports_(
                f"from {parent_package}.nvrtc import",
                hiprtc_generator.backend.walk_entities_to_import(False),
            ),
            extra_cimports=collect_imports_(
                f"from {parent_package}.nvrtc cimport",
                hiprtc_generator.backend.walk_entities_to_cimport(False),
            ),
            extra_cmodule_cimports=collect_imports_(
                f"from {parent_package}.{cmodule_prefix}nvrtc cimport",
                hiprtc_generator.backend.walk_entities_to_cimport(True),
            ),
            cuda_cmodule_prefix=cmodule_prefix,
        )

        # writer driver module files
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            output_dir,
            f"{parent_package}.{config['driver_name']}",
            hip_generator,
            hip_2_cuda,
            license_text,
            **cuda_extra_args,
        )

        # write runtime module files
        cuda_interop_layer_gen.generate_cuda_interop_module_files(
            output_dir,
            f"{parent_package}.{config['runtime_name']}",
            hip_generator,
            hip_2_cuda,
            license_text,
            warn=False,
            **cuda_extra_args,
        )  # NOTE: cudart is the same as cuda, but we generate it to have also the corresponding pxd/pyx files. Could be solved via symlinks & __init__.py mod too.


# Per-library dispatch. Each entry is (callable, package-shortname).
# The callables live in the per-package modules `generators_hip`,
# `generators_libraries`, and `generators_systems`; they take all required state as
# keyword-only arguments, so this orchestrator does not need module-
# level globals to feed them. Package shortname routes the generator
# output into the right `packages/rocm-bindings-<pkg>/...` directory.
AVAILABLE_GENERATORS = {
    # rocm-bindings-hip
    "hip":       (generators_hip.generate_hip,             "hip",       "hip/hip_runtime.h"),
    "hiprtc":    (generators_hip.generate_hiprtc,          "hip",       "hip/hiprtc.h"),
    # rocm-bindings-systems
    "rccl":      (generators_systems.generate_rccl,        "systems",   "rccl/rccl.h"),
    "roctx":     (generators_systems.generate_roctx,       "systems",   "roctracer/roctx.h"),
    "hipfile":   (generators_systems.generate_hipfile,     "systems",   "hipfile.h"),
    "amdsmi":    (generators_systems.generate_amdsmi,      "systems",   "amd_smi/amdsmi.h"),
    "hsa":       (generators_systems.generate_hsa,         "systems",   "hsa/hsa_ext_amd.h"),
    # rocm-bindings-libraries
    "hipblas":     (generators_libraries.generate_hipblas,     "libraries", "hipblas/hipblas.h"),
    "hipblaslt":   (generators_libraries.generate_hipblaslt,   "libraries", "hipblaslt/hipblaslt.h"),
    "hiprand":     (generators_libraries.generate_hiprand,     "libraries", "hiprand/hiprand.h"),
    "hipfft":      (generators_libraries.generate_hipfft,      "libraries", "hipfft/hipfft.h"),
    "hipsparse":   (generators_libraries.generate_hipsparse,   "libraries", "hipsparse/hipsparse.h"),
    "hipsparselt": (generators_libraries.generate_hipsparselt, "libraries", "hipsparselt/hipsparselt.h"),
    "hipsolver":   (generators_libraries.generate_hipsolver,   "libraries", "hipsolver/hipsolver.h"),
    "hiptensor":   (generators_libraries.generate_hiptensor,   "libraries", "hiptensor/hiptensor.h"),
    "hipdnn":      (generators_libraries.generate_hipdnn,      "libraries", "hipdnn_backend.h"),
    # rocm-bindings-compiler — both formerly their own recipes; now
    # libraries inside the hip recipe. amd_comgr is single-header;
    # llvm is multi-module (header_relpath=None signals the orchestrator
    # to pass output_dir instead of resolving a single header).
    "amd_comgr": (generators_compiler.generate_amd_comgr,  "compiler",  "amd_comgr/amd_comgr.h"),
    "llvm":      (generators_compiler.write_llvm_modules,  "compiler",  None),
}

_PKG_TO_DIR = {
    "hip":       ("rocm-bindings-hip",       "src", "rocm", "bindings"),
    "libraries": ("rocm-bindings-libraries", "src", "rocm", "bindings"),
    "systems":   ("rocm-bindings-systems",   "src", "rocm", "bindings"),
    "compiler":  ("rocm-bindings-compiler",  "src", "rocm", "bindings"),
}


def generate(opts):  # noqa: C901
    """Run HIP + interop subgenerators against the modern hip-python layout.

    Writes only `.pxd`/`.pyx` files. Python-packaging artifacts (`__init__.py`,
    `_version.py.in`, `setup.py`, `requirements.txt`, docs) are NOT produced
    — those are handcoded in the hip-python repo.

    `opts` is the unified-CLI options object. Required attributes:
      output_dir, rocm_version, clang_resource_dir, runtime_linking,
      include, exclude.
    Optional: rocm_path, rocm_systems_dir, rocm_libraries_dir,
      rocm_llvm_project_dir.
    """
    output_dir = opts.output_dir
    # Support both rocm_path and repository directories
    rocm_inc = os.path.join(opts.rocm_path, "include") if opts.rocm_path else None
    rocm_systems_dir = opts.rocm_systems_dir
    rocm_libraries_dir = opts.rocm_libraries_dir
    rocm_llvm_project_dir = opts.rocm_llvm_project_dir
    runtime_linking = opts.runtime_linking

    # hipify-perl is optional now (only needed for CUDA interop)
    if opts.rocm_path:
        hipify_perl_path = os.path.join(opts.rocm_path, "bin", "hipify-perl")
        (_, hip_2_cuda) = parse_hipify_perl(hipify_perl_path)
    else:
        hip_2_cuda = {}  # Empty if no CUDA interop

    rocm_v = opts.rocm_version.split(".")
    rocm_version_tuple = (int(rocm_v[0]), int(rocm_v[1]), int(rocm_v[2]))
    # Populated by generators_hip.generate_hip when the 'hip' library is in lib_names.
    hip_version_tuple = (0, 0, 0, "")

    # Build include paths from all available sources (ROCM_PATH + repositories).
    # The codegen always targets AMD; no nvidia path is exercised in practice.
    generator_args = list(opts.generator_args or [])
    generator_args += build_generator_include_paths(
        rocm_inc, rocm_systems_dir, rocm_libraries_dir, rocm_llvm_project_dir,
    )

    if not opts.clang_resource_dir:
        raise RuntimeError(
            "Clang resource directory is not set. Pass --clang-resource-dir "
            "(e.g. `$($ROCM_PATH/llvm/bin/clang -print-resource-dir)`)."
        )
    generator_args += ["-resource-dir", opts.clang_resource_dir]

    # Shared kwargs every per-package callable accepts. Per-callable
    # extras (include_dir/header_relpath/header_content per library,
    # rocm_version_tuple for hiprtc) are added at dispatch time.
    common_kwargs = dict(
        runtime_linking=runtime_linking,
        generator_args=generator_args,
        default_ptr_handler=HIP_PYTHON_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
    )

    # Resolve the requested library set from the wheel-level --include /
    # --exclude flags. Each AVAILABLE_GENERATORS entry maps to a wheel via
    # its `pkg_short` (the second tuple element); a library is selected
    # when its wheel is in the (include - exclude) set.
    selected_wheels = set(opts.include) - set(opts.exclude)
    lib_names = [
        name for name, (_callable, pkg_short, _relpath) in AVAILABLE_GENERATORS.items()
        if pkg_short in selected_wheels
    ]

    # Pre-create the per-package output dirs so write_module_files can
    # drop files in straight away.
    pkg_dirs = {
        pkg_short: os.path.join(output_dir, "packages", *path_parts)
        for pkg_short, path_parts in _PKG_TO_DIR.items()
    }
    cuda_output_dir = os.path.join(
        output_dir, "packages", "hip-python-interop", "src", "cuda"
    )
    for d in pkg_dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cuda_output_dir, "bindings")).mkdir(parents=True, exist_ok=True)

    # The cuda interop layer needs the hip + hiprtc generators to wire
    # up its alias bindings. We capture them as we walk the lib list so
    # they're available without relying on module-level globals.
    hip_generator = None
    hiprtc_generator = None

    # Validate all requested libraries upfront before kicking off work.
    for libname in lib_names:
        if libname not in AVAILABLE_GENERATORS:
            avail = ", ".join(f"'{a}'" for a in avail_lib_names)
            raise KeyError(
                f"no codegenerator found for library '{libname}'; please choose "
                f"from: {avail}, or '*'."
            )

    # Split work: hip + hiprtc must run in the main process because their
    # generator objects (libclang AST cursors) are needed by the cuda interop
    # layer and aren't safely picklable across processes. Everything else
    # is fully self-contained — write_module_files() persists output to disk.
    SEQUENTIAL_LIBS = {"hip", "hiprtc"}
    sequential_libs = [n for n in lib_names if n in SEQUENTIAL_LIBS]
    parallel_libs = [n for n in lib_names if n not in SEQUENTIAL_LIBS]

    # Per-library log files land in a unique /tmp directory so concurrent
    # runs (or repeat runs) don't clobber each other.
    log_dir = tempfile.mkdtemp(prefix=f"hip_python_codegen_{os.getpid()}_")
    log_paths = {}
    errors = {}
    multi_module_names = {}  # libname -> list[str] (for llvm and other multi-module libs)

    print(f"[info] per-library logs: {log_dir}", file=sys.stderr)

    def _run_sequential(libname):
        """Run one library in the main process. Captures hip/hiprtc generators."""
        nonlocal hip_generator, hiprtc_generator, hip_version_tuple
        log_path = os.path.join(log_dir, f"{libname}.log")
        log_paths[libname] = log_path
        print(f"[start] {libname}", file=sys.stderr)
        start = time.time()
        try:
            callable_, pkg_short, header_relpath = AVAILABLE_GENERATORS[libname]
            if header_relpath is None:
                # Multi-module library (currently: llvm). Callable owns
                # both the build and the writing.
                if not rocm_inc:
                    print(
                        f"[skip] {libname}: no --rocm-path; cannot derive llvm/include",
                        file=sys.stderr,
                    )
                    return
                include_dir = os.path.join(os.path.dirname(rocm_inc), "llvm", "include")
                kwargs = dict(common_kwargs)
                kwargs["output_dir"] = pkg_dirs[pkg_short]
                kwargs["include_dir"] = include_dir
                module_names = callable_(**kwargs)
                if module_names:
                    multi_module_names[libname] = module_names
                elapsed = time.time() - start
                print(f"[done] {libname} ({elapsed:.1f}s)", file=sys.stderr)
                return
            try:
                header_path, header_content = resolve_header_path(
                    header_relpath, rocm_inc, rocm_systems_dir,
                    rocm_libraries_dir, rocm_llvm_project_dir,
                )
            except FileNotFoundError as e:
                _log.warning(f"Skipping '{libname}': {e}")
                print(f"[skip] {libname}: header not found", file=sys.stderr)
                return
            include_dir = _resolve_include_dir(header_path, header_relpath)
            kwargs = dict(common_kwargs)
            kwargs["include_dir"] = include_dir
            kwargs["header_relpath"] = header_relpath
            if header_content is not None:
                kwargs["header_content"] = header_content
            if libname == "hiprtc":
                kwargs["rocm_version_tuple"] = rocm_version_tuple
            result = callable_(**kwargs)
            if libname == "hip":
                generator, hip_version_tuple = result
                hip_generator = generator
            else:
                generator = result
                if libname == "hiprtc":
                    hiprtc_generator = generator
            generator.write_module_files(output_dir=pkg_dirs[pkg_short])
            elapsed = time.time() - start
            print(f"[done] {libname} ({elapsed:.1f}s)", file=sys.stderr)
        except Exception as e:
            elapsed = time.time() - start
            errors[libname] = str(e)
            print(f"[error] {libname} ({elapsed:.1f}s): {e}", file=sys.stderr)
            raise

    if parallel_libs:
        # Submit parallel work first, then run sequential libs concurrently
        # in the main process. Pool workers carry their own log file handle.
        with ProcessPoolExecutor() as pool:
            futures = {}
            for libname in parallel_libs:
                log_path = os.path.join(log_dir, f"{libname}.log")
                log_paths[libname] = log_path
                print(f"[start] {libname}", file=sys.stderr)
                future = pool.submit(
                    _worker_generate_library,
                    libname, output_dir, rocm_inc, rocm_systems_dir,
                    rocm_libraries_dir, rocm_llvm_project_dir,
                    runtime_linking, generator_args, rocm_version_tuple,
                    log_path,
                )
                futures[future] = libname

            # Sequential libs run in this process while the pool is busy.
            for libname in sequential_libs:
                _run_sequential(libname)

            # Drain pool results.
            for future in as_completed(futures):
                libname = futures[future]
                try:
                    (lib, _lp, elapsed, status, mod_names, err) = future.result()
                except Exception as e:
                    errors[libname] = str(e)
                    print(f"[error] {libname}: {e}", file=sys.stderr)
                    continue
                if status == "ok":
                    if mod_names:
                        multi_module_names[lib] = mod_names
                    print(f"[done] {lib} ({elapsed:.1f}s)", file=sys.stderr)
                else:
                    errors[lib] = err
                    print(f"[error] {lib} ({elapsed:.1f}s): {err}", file=sys.stderr)
    else:
        # Nothing to parallelize — run everything sequentially.
        for libname in sequential_libs:
            _run_sequential(libname)

    # CUDA interop layer (writes into <output_dir>/packages/hip-python-interop/cuda/...)
    license_path = opts.license_path or os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "LICENSE"
    )
    with open(license_path, "r") as licensefile:
        license_text = "".join(
            f"# {ln}\n" for ln in licensefile.read().rstrip().splitlines()
        )
    generate_cuda_interop_layer_files(
        license_text=license_text,
        output_dir=output_dir,
        hip_generator=hip_generator,
        hiprtc_generator=hiprtc_generator,
        hip_2_cuda=hip_2_cuda,
        rocm_version_tuple=rocm_version_tuple,
    )

    # Return data the orchestrator may want (e.g. version metadata for
    # cmake/generated_versions.cmake; per-package module lists).
    libraries_modules = [
        n for n in lib_names if AVAILABLE_GENERATORS[n][1] == "libraries"
    ]
    systems_modules = [
        n for n in lib_names if AVAILABLE_GENERATORS[n][1] == "systems"
    ]
    compiler_modules = [
        n for n in lib_names if AVAILABLE_GENERATORS[n][1] == "compiler"
    ]
    # Multi-module libraries (currently: llvm) emit several Cython
    # modules per AVAILABLE_GENERATORS entry. Their dotted names are
    # surfaced via the worker's `module_names` return value, collected
    # into multi_module_names. Codegen.py reads `llvm_modules` to
    # bucket entries into HIP_PYTHON_LLVM_C_MODULES, etc.
    llvm_modules = multi_module_names.get("llvm", [])
    return dict(
        rocm_version=rocm_version_tuple,
        hip_version=hip_version_tuple,
        hip_modules=lib_names,
        libraries_modules=libraries_modules,
        systems_modules=systems_modules,
        compiler_modules=compiler_modules,
        llvm_modules=llvm_modules,
        log_paths=log_paths,
        errors=errors,
    )


# ==============================================================================
# Cross-cutting build-system inputs (Cython namespace markers + cmake lists)
# ==============================================================================
# These writers describe the binding artifacts produced by `generate(opts)` to
# downstream consumers (Cython compiler, cmake). They live alongside the
# orchestrator because the same module that knows what was generated should be
# the one that emits the cross-cutting outputs.

from interfacegen.support import gitversion

NAMESPACE_MARKER_BODY = (
    "# Cython namespace package marker — build-time only, not installed.\n"
    "# Auto-generated by the hip-python code generator.\n"
)

_AUTOGEN_HEADER = (
    "# AUTO-GENERATED by the hip-python code generator. Do not edit by hand.\n"
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
        os.path.join(opts.output_dir, "packages", "rocm-bindings-hip", "src", "rocm", "bindings"),
        os.path.join(opts.output_dir, "packages", "rocm-bindings-libraries", "src", "rocm", "bindings"),
        os.path.join(opts.output_dir, "packages", "rocm-bindings-systems", "src", "rocm", "bindings"),
        os.path.join(opts.output_dir, "packages", "rocm-bindings-compiler", "src", "rocm", "bindings"),
        os.path.join(opts.output_dir, "packages", "hip-python-interop", "src", "cuda", "bindings"),
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
      - compiler:   union of `llvm_modules` and `compiler_modules`,
                    both produced by the HIP recipe. `compiler_modules`
                    lists single-header libraries under the "compiler"
                    package (currently just `amd_comgr`); `llvm_modules`
                    lists the dotted names of every Cython module emitted
                    by the LLVM multi-module library.
    """
    # rocm-bindings-libraries
    hip_result = recipe_results.get("hip")
    if hip_result is not None:
        libs = hip_result.get("libraries_modules") or []
        path = os.path.join(
            opts.output_dir, "packages", "rocm-bindings-libraries", "cmake",
            "generated_modules.cmake",
        )
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(_AUTOGEN_HEADER)
            f.write(f"set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES\n    {' '.join(libs)})\n")

    # rocm-bindings-systems
    if hip_result is not None:
        sys_libs = hip_result.get("systems_modules") or []
        path = os.path.join(
            opts.output_dir, "packages", "rocm-bindings-systems", "cmake",
            "generated_modules.cmake",
        )
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(_AUTOGEN_HEADER)
            f.write(f"set(HIP_PYTHON_SYSTEMS_GENERATED_MODULES\n    {' '.join(sys_libs)})\n")

    # rocm-bindings-compiler
    # llvm modules now flow through the hip recipe (multi-module library
    # registered in `binding_generator.AVAILABLE_GENERATORS`).
    llvm_modules = recipe_results.get("hip", {}).get("llvm_modules") or []
    # comgr modules now flow through the hip recipe (single-header library
    # registered in `binding_generator.AVAILABLE_GENERATORS`). The hip result
    # exposes them via `compiler_modules` (e.g., ["amd_comgr"]).
    comgr_modules = recipe_results.get("hip", {}).get("compiler_modules") or []
    if llvm_modules or comgr_modules:
        path = os.path.join(
            opts.output_dir, "packages", "rocm-bindings-compiler", "cmake",
            "generated_modules.cmake",
        )
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        # Subdivide LLVM modules by their on-disk subdirectory.
        c_modules, transforms_modules, config_modules = [], [], []
        compiler_root = os.path.join(
            opts.output_dir, "packages", "rocm-bindings-compiler", "rocm", "bindings", "llvm",
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


def write_version_template_file(opts, recipe_results):
    """Write `<output_dir>/VERSION.in` consumed by packages/CMakeLists.txt.

    The template embeds the ROCm version and the codegen tool's own
    rev-count; `@HIP_PYTHON_VERSION_SHORT@` is filled in at consumer
    cmake-configure time from the consumer repo's `git rev-list --count
    HEAD`. Format: `<rocm_version>.<codegen_rev_count>.@HIP_PYTHON_VERSION_SHORT@`.

    `VERSION.in` (and the `VERSION` it renders to) MUST NOT be committed
    on the codegen base branch; both are gitignored and emitted fresh by
    every codegen run.
    """
    try:
        codegen_rev_count = gitversion.git_head_rev_count()
    except Exception:
        codegen_rev_count = "0"
    body = f"{opts.rocm_version}.{codegen_rev_count}.@HIP_PYTHON_VERSION_SHORT@"
    path = os.path.join(opts.output_dir, "VERSION.in")
    with open(path, "w") as f:
        f.write(body)


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

    # Capture the upstream source-tree commits that contributed
    # headers to this codegen run. Each is optional — when the
    # caller didn't pass `--rocm-{systems,libraries,llvm-project}-dir`
    # the corresponding field is empty (rendered as
    # "*not consulted*" by the docs landing page).
    def _git_head(path):
        if not path:
            return ""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=path,
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    rocm_libraries_rev = _git_head(getattr(opts, "rocm_libraries_dir", None))
    rocm_systems_rev = _git_head(getattr(opts, "rocm_systems_dir", None))
    rocm_llvm_project_rev = _git_head(
        getattr(opts, "rocm_llvm_project_dir", None)
    )

    # ISO-8601 UTC; pinned at codegen time so the docs landing page
    # can show "generated on <DATE>" rather than the sphinx-build
    # date (which would change every time someone rebuilds docs).
    codegen_date = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0).isoformat()
    )

    body = (
        _AUTOGEN_HEADER
        + f'set(HIP_PYTHON_GENERATED_ROCM_VERSION         "{rocm_version}")\n'
        + f'set(HIP_PYTHON_GENERATED_HIP_VERSION          "{hip_version_str}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_BRANCH       "{codegen_branch}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_REV          "{codegen_rev}")\n'
        + f'set(HIP_PYTHON_GENERATED_CODEGEN_VERSION      "{codegen_version}")\n'
        + f'set(HIP_PYTHON_GENERATED_DATE                 "{codegen_date}")\n'
        + f'set(HIP_PYTHON_GENERATED_ROCM_LIBRARIES_REV   "{rocm_libraries_rev}")\n'
        + f'set(HIP_PYTHON_GENERATED_ROCM_SYSTEMS_REV     "{rocm_systems_rev}")\n'
        + f'set(HIP_PYTHON_GENERATED_ROCM_LLVM_PROJECT_REV "{rocm_llvm_project_rev}")\n'
    )
    for pkg in ("rocm-bindings-hip", "rocm-bindings-libraries",
                "rocm-bindings-systems", "rocm-bindings-compiler",
                "hip-python-interop"):
        path = os.path.join(opts.output_dir, "packages", pkg, "cmake", "generated_versions.cmake")
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(body)


