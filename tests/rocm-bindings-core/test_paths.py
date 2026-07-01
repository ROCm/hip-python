# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Tests for ``rocm.bindings.util.paths.get_library_path`` resolution of the
LLVM toolchain libraries (``clang`` / ``LLVM``).

These lock down the ``ROCM_PATH`` / ``ROCM_HOME`` env-var tier added for the
LLVM toolchain:

  * ``ROCM_HOME`` is honored (in addition to ``ROCM_PATH``).
  * ``clang`` / ``LLVM`` resolve under ``<rocm>/llvm/lib``.
  * The versioned soname (e.g. ``libclang.so.23.0git``) is matched via glob
    when the bare ``libclang.so`` symlink is absent (as in rocm-sdk-core).
  * ``ROCM_PATH`` takes precedence over ``ROCM_HOME``.

``rocm_sdk`` is blocked in these tests so the wheel-install tiers (which are
verified separately against real venvs) do not shadow the env-var tier.
"""

import sys
import types

import pytest

from rocm.bindings.util import paths


@pytest.fixture
def block_rocm_sdk(monkeypatch):
    """Make ``import rocm_sdk`` (and submodules) fail, forcing the env tier."""
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)


def _make_llvm_lib(root):
    llvm_lib = root / "llvm" / "lib"
    llvm_lib.mkdir(parents=True)
    return llvm_lib


# These tests target the ROCM_PATH / ROCM_HOME (tier 3) resolution, so they
# must be isolated from tier-1 bundled auto-detection: get_library_path()
# otherwise rglobs the installed rocm package and, in a wheel install, finds
# the bundled libLLVM.so, shadowing the env-var tier. Passing an explicit
# bundled_location that does not contain the library disables the rglob.
def _no_bundled(tmp_path):
    return tmp_path / "no_bundled"


def test_get_library_path_clang_rocm_home(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libclang.so").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path(
        "clang", bundled_location=_no_bundled(tmp_path)
    ) == str(llvm_lib / "libclang.so").encode("utf-8")


def test_get_library_path_clang_versioned_soname(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    # Only the versioned soname is present (no bare libclang.so), as shipped by
    # rocm-sdk-core.
    (llvm_lib / "libclang.so.23.0git").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path(
        "clang", bundled_location=_no_bundled(tmp_path)
    ).decode("utf-8") == str(llvm_lib / "libclang.so.23.0git")


def test_get_library_path_llvm_rocm_path(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libLLVM.so.23.0git").touch()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))

    assert paths.get_library_path(
        "LLVM", bundled_location=_no_bundled(tmp_path)
    ).decode("utf-8") == str(llvm_lib / "libLLVM.so.23.0git")


def test_get_library_path_rocm_path_wins_over_rocm_home(
    tmp_path, monkeypatch, block_rocm_sdk
):
    # libclang only under the ROCM_PATH tree; ROCM_HOME points elsewhere.
    path_tree = tmp_path / "path_tree"
    home_tree = tmp_path / "home_tree"
    llvm_lib = _make_llvm_lib(path_tree)
    (llvm_lib / "libclang.so").touch()
    _make_llvm_lib(home_tree)  # exists but has no libclang

    monkeypatch.setenv("ROCM_PATH", str(path_tree))
    monkeypatch.setenv("ROCM_HOME", str(home_tree))

    assert paths.get_library_path(
        "clang", bundled_location=_no_bundled(tmp_path)
    ) == str(llvm_lib / "libclang.so").encode("utf-8")


def test_get_library_path_rocm_sdk_returns_pathlib(tmp_path, monkeypatch):
    """Regression: ``rocm_sdk.find_libraries`` returns ``pathlib.Path`` objects.

    The tier-2 general branch used to call ``.encode()`` directly on the Path,
    raising ``AttributeError`` that was silently swallowed, so every rocm_sdk
    runtime lib fell through to ``/opt/rocm`` (masking) or a bare soname (which
    fails in a pip-only install). ``get_library_path`` must return the resolved
    absolute path as bytes regardless of whether ``find_libraries`` yields a
    ``Path`` or a ``str``.
    """
    lib_file = tmp_path / "libamdhip64.so"
    lib_file.touch()

    fake_rocm_sdk = types.ModuleType("rocm_sdk")
    fake_rocm_sdk.find_libraries = lambda shortname: [lib_file]  # PosixPath
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_rocm_sdk)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)

    # bundled_location points at an empty dir so the tier-1 auto-detect rglob
    # (which would scan the installed rocm package) is skipped and the rocm_sdk
    # tier is exercised deterministically.
    result = paths.get_library_path("amdhip64", bundled_location=tmp_path / "empty")

    assert result == str(lib_file).encode("utf-8")


def test_get_library_path_rocm_sdk_returns_str(tmp_path, monkeypatch):
    """The rocm_sdk tier also accepts plain ``str`` paths (forward-compat)."""
    lib_file = tmp_path / "libamdhip64.so"
    lib_file.touch()

    fake_rocm_sdk = types.ModuleType("rocm_sdk")
    fake_rocm_sdk.find_libraries = lambda shortname: [str(lib_file)]
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_rocm_sdk)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)

    result = paths.get_library_path("amdhip64", bundled_location=tmp_path / "empty")

    assert result == str(lib_file).encode("utf-8")
