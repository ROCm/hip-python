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


def test_get_library_path_clang_rocm_home(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libclang.so").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path("clang") == str(
        llvm_lib / "libclang.so"
    ).encode("utf-8")


def test_get_library_path_clang_versioned_soname(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    # Only the versioned soname is present (no bare libclang.so), as shipped by
    # rocm-sdk-core.
    (llvm_lib / "libclang.so.23.0git").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path("clang").decode("utf-8") == str(
        llvm_lib / "libclang.so.23.0git"
    )


def test_get_library_path_llvm_rocm_path(tmp_path, monkeypatch, block_rocm_sdk):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libLLVM.so.23.0git").touch()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))

    assert paths.get_library_path("LLVM").decode("utf-8") == str(
        llvm_lib / "libLLVM.so.23.0git"
    )


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

    assert paths.get_library_path("clang") == str(
        llvm_lib / "libclang.so"
    ).encode("utf-8")
