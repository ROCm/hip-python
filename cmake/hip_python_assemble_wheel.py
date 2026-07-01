#!/usr/bin/env python3
"""Assemble a hip-python wheel from the unified build's compiled output.

This replaces the per-package ``python -m build --wheel`` invocation in
the unified ``all_wheels`` flow. ``python -m build`` would re-run
scikit-build-core, which re-compiles every Cython extension a second
time (once in the unified tree, once in scikit-build-core's own
per-package build tree -- a different CMake source root, so the work
cannot be shared). The unified build has already compiled every
extension; this script collects that output and packs it into a wheel
that is equivalent to what scikit-build-core would have produced, so
each module is compiled exactly once.

A scikit-build-core wheel for these packages is the union of two
sources:

1. The CMake install component (``install.components`` in the package
   pyproject) -- the compiled ``.so`` modules plus the explicitly
   ``install()``-ed ``.pxd`` / ``.pyi`` / ``.py`` / generated
   ``_version.py`` files. Reproduced here via ``cmake --install
   <unified-build> --component <component> --prefix <staging>``, which
   also applies each target's ``INSTALL_RPATH`` exactly as
   scikit-build-core does.
2. The ``wheel.packages`` source overlay (e.g. ``src/rocm`` ->
   ``rocm/``) -- pure-Python modules, ``.pyx`` sources, ``.pyi`` stubs,
   namespace markers, and any other source files not produced by the
   CMake install. Reproduced here by copying the source subtree on top
   of the install staging dir.

The ``.dist-info`` metadata (``METADATA``) is generated with
``pyproject-metadata`` from the package's ``pyproject.toml`` -- the same
PEP 621 -> core-metadata path scikit-build-core uses -- with the dynamic
version filled in from the per-package ``VERSION`` file. ``RECORD`` and
the zip are written directly (no dependency on the ``wheel`` CLI).

The emitted wheel carries a generic ``linux_<arch>`` platform tag; the
unified build's ``hip_python_add_wheel_target`` tail then runs
``auditwheel repair`` (when enabled), which retags it to the
``manylinux_*`` tags -- identical to the existing flow.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - 3.9 / 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import pyproject_metadata
from packaging.utils import canonicalize_name
from packaging.version import Version


def _log(message: str) -> None:
    print(f"[assemble-wheel] {message}", flush=True)


def read_version(package_dir: Path, override: str | None) -> str:
    if override:
        return override.strip()
    version_file = package_dir / "VERSION"
    if not version_file.is_file():
        raise SystemExit(
            f"VERSION file not found at {version_file}. The unified configure "
            "step populates it; run `cmake -B build -S packages` first."
        )
    return version_file.read_text(encoding="utf-8").strip()


def read_wheel_package_dirs(pyproject: dict) -> list[str]:
    """Source subdirs scikit-build-core overlays onto the wheel.

    Mirrors ``tool.scikit-build.wheel.packages``; defaults to
    ``["src/rocm"]`` to match the binding packages.
    """
    skb = pyproject.get("tool", {}).get("scikit-build", {})
    packages = skb.get("wheel", {}).get("packages")
    if packages:
        return list(packages)
    return ["src/rocm"]


def compute_platform_tag(abi3_floor: str | None = None) -> str:
    """Pre-repair interpreter/abi/platform tag, e.g. cp312-cp312-linux_x86_64.

    The platform component is the generic (non-manylinux) tag, matching
    what scikit-build-core emits before ``auditwheel repair`` retags the
    wheel to its ``manylinux_*`` aliases.

    When ``abi3_floor`` is given (e.g. ``"3.9"``), the modules were built
    against the CPython stable ABI, so the tag becomes
    ``cp<floor>-abi3-<platform>`` (a single wheel that loads on every
    CPython >= the floor) instead of the full interpreter-specific tag.
    """
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if abi3_floor:
        major, minor = abi3_floor.split(".")[:2]
        interp = f"cp{major}{minor}"
        return f"{interp}-abi3-{platform}"
    interp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # Standard CPython extension modules use the full interpreter ABI
    # (the EXT_SUFFIX SOABI starts with "cpython-<ver>"), so the abi tag
    # equals the interpreter tag.
    abi = interp
    return f"{interp}-{abi}-{platform}"


def cmake_install_component(
    cmake: str, build_dir: Path, component: str, staging: Path, config: str
) -> None:
    """Install one CMake component from the unified build into staging.

    Relative ``install(DESTINATION ...)`` paths land under
    ``staging/`` (the wheel root), and ``INSTALL_RPATH`` is applied to
    the compiled modules -- exactly the tree scikit-build-core would
    install.
    """
    cmd = [
        cmake,
        "--install",
        str(build_dir),
        "--component",
        component,
        "--prefix",
        str(staging),
        "--config",
        config,
    ]
    _log(f"cmake --install component '{component}' -> {staging}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            f"cmake --install failed for component '{component}' "
            f"(exit {result.returncode})."
        )


def overlay_source_tree(package_dir: Path, subdir: str, staging: Path) -> None:
    """Copy a wheel.packages source subtree onto the staging dir.

    ``src/rocm`` -> ``staging/rocm``. Files already present from the
    CMake install (compiled ``.so``, generated ``_version.py``) are
    authoritative and left untouched; only source-only files (``.pyx``,
    ``.pyi``, pure ``.py``, namespace markers) are added. ``__pycache__``
    and ``.pyc`` are skipped, matching scikit-build-core.
    """
    source_root = package_dir / subdir
    if not source_root.is_dir():
        raise SystemExit(f"wheel.packages source dir not found: {source_root}")
    dest_root = staging / source_root.name
    for src in sorted(source_root.rglob("*")):
        if "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = src.relative_to(source_root)
        dest = dest_root / rel
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if dest.exists():
            # Installed file wins (identical content for shared .pxd/.pyi;
            # authoritative for generated files).
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def build_metadata(package_dir: Path, pyproject: dict, version: str) -> str:
    md = pyproject_metadata.StandardMetadata.from_pyproject(
        pyproject, project_dir=package_dir
    )
    md.version = Version(version)
    md.dynamic = [d for d in md.dynamic if d != "version"]
    return bytes(md.as_rfc822()).decode("utf-8")


def license_file_paths(package_dir: Path, pyproject: dict) -> list[Path]:
    project = pyproject.get("project", {})
    patterns = project.get("license-files")
    if patterns is None:
        # PEP 639 default when license-files is omitted.
        patterns = ["LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*"]
    found: list[Path] = []
    for pattern in patterns:
        for path in sorted(package_dir.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def write_dist_info(
    staging: Path,
    dist_name: str,
    version: str,
    metadata_text: str,
    tag: str,
    license_files: list[Path],
    package_dir: Path,
) -> Path:
    dist_info = staging / f"{dist_name}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    (dist_info / "METADATA").write_text(metadata_text, encoding="utf-8")

    wheel_text = (
        "Wheel-Version: 1.0\n"
        "Generator: hip_python_assemble_wheel\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    )
    (dist_info / "WHEEL").write_text(wheel_text, encoding="utf-8")

    if license_files:
        licenses_dir = dist_info / "licenses"
        for path in license_files:
            rel = path.relative_to(package_dir)
            dest = licenses_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

    return dist_info


def _record_hash(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii"), len(data)


def pack_wheel(staging: Path, dist_info: Path, output: Path) -> None:
    """Write RECORD and zip the staging tree into a wheel.

    File order is sorted for determinism; RECORD is written last with an
    empty hash/size for its own entry (per the wheel spec).
    """
    record_path = dist_info / "RECORD"

    # Hash every payload file (RECORD itself does not exist yet, so it is
    # naturally excluded from its own hashed entries).
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    record_lines: list[str] = []
    for path in payload:
        arcname = path.relative_to(staging).as_posix()
        digest, size = _record_hash(path)
        record_lines.append(f"{arcname},{digest},{size}")
    record_arcname = record_path.relative_to(staging).as_posix()
    record_lines.append(f"{record_arcname},,")
    record_path.write_text("\n".join(record_lines) + "\n", encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    # Now include RECORD in the archive alongside the payload.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in payload + [record_path]:
            zf.write(path, path.relative_to(staging).as_posix())
    _log(f"wrote {output}")


def assemble(
    cmake: str,
    build_dir: Path,
    component: str,
    package_dir: Path,
    output_dir: Path,
    config: str,
    version_override: str | None,
    tag_override: str | None,
    abi3_floor: str | None,
) -> Path:
    pyproject = tomllib.loads((package_dir / "pyproject.toml").read_text("utf-8"))
    project_name = pyproject["project"]["name"]
    dist_name = canonicalize_name(project_name).replace("-", "_")
    version = read_version(package_dir, version_override)
    tag = tag_override or compute_platform_tag(abi3_floor)
    wheel_pkg_dirs = read_wheel_package_dirs(pyproject)

    with tempfile.TemporaryDirectory(prefix=f"{dist_name}-wheel-") as tmp:
        staging = Path(tmp) / "wheel"
        staging.mkdir(parents=True)

        cmake_install_component(cmake, build_dir, component, staging, config)
        for subdir in wheel_pkg_dirs:
            overlay_source_tree(package_dir, subdir, staging)

        metadata_text = build_metadata(package_dir, pyproject, version)
        licenses = license_file_paths(package_dir, pyproject)
        dist_info = write_dist_info(
            staging, dist_name, version, metadata_text, tag, licenses, package_dir
        )

        output = output_dir / f"{dist_name}-{version}-{tag}.whl"
        pack_wheel(staging, dist_info, output)
        return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", default="cmake", help="cmake executable")
    parser.add_argument(
        "--build-dir", required=True, help="unified CMake build directory"
    )
    parser.add_argument(
        "--component", required=True, help="CMake install component to collect"
    )
    parser.add_argument(
        "--package-dir", required=True, help="package source dir (has pyproject.toml)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory to write the wheel into"
    )
    parser.add_argument("--config", default="Release", help="CMake install config")
    parser.add_argument(
        "--version", default=None, help="override version (default: read VERSION)"
    )
    parser.add_argument(
        "--tag", default=None, help="override wheel tag (default: computed)"
    )
    parser.add_argument(
        "--abi3-floor",
        default=None,
        help=(
            "CPython stable-ABI floor (e.g. 3.9); tags the wheel "
            "cp<floor>-abi3. Ignored when --tag is given."
        ),
    )
    args = parser.parse_args(argv)

    assemble(
        cmake=args.cmake,
        build_dir=Path(args.build_dir).resolve(),
        component=args.component,
        package_dir=Path(args.package_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        config=args.config,
        version_override=args.version,
        tag_override=args.tag,
        abi3_floor=args.abi3_floor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
