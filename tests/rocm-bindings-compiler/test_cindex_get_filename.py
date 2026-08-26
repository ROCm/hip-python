# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression tests for ``Config.get_filename`` in
``rocm.bindings.clang.cindex``.

When neither an explicit ``Config.set_library_file``/``set_library_path`` nor
the ``LIBCLANG_LIBRARY_FILE``/``LIBCLANG_LIBRARY_PATH`` env vars are set, the
loader used to return the bare ``libclang.so`` soname and rely on the system
loader, which fails in a pip-only ROCm install. It must instead fall back to
the shared ``rocm.bindings.util.paths.get_library_path('clang')`` resolver
(the same path the rest of the bindings use).

Explicit ``Config`` / ``LIBCLANG_*`` overrides must keep priority over the
resolver fallback.
"""

import platform

import pytest
from rocm.bindings.clang import cindex
from rocm.bindings.util import paths as _paths


@pytest.fixture
def clear_config(monkeypatch):
    """Reset the process-wide Config overrides for a deterministic default."""
    monkeypatch.setattr(cindex.Config, "library_file", None, raising=False)
    monkeypatch.setattr(cindex.Config, "library_path", None, raising=False)


def test_get_filename_falls_back_to_resolver(
    tmp_path, monkeypatch, clear_config
):
    lib_file = tmp_path / "libclang.so.19.1"
    lib_file.touch()

    monkeypatch.setattr(
        _paths,
        "get_library_path",
        lambda shortname: str(lib_file).encode("utf-8"),
    )

    assert cindex.Config().get_filename() == str(lib_file)


def test_get_filename_ignores_resolver_when_not_absolute(
    tmp_path, monkeypatch, clear_config
):
    # A bare soname (resolver's own fallback) must not be returned as if it
    # were a resolved path; get_filename keeps its bare-name default instead.
    # That default is per-platform, so assert against the platform's own
    # spelling rather than the Linux one.
    expected = {
        "Windows": "libclang.dll",
        "Darwin": "libclang.dylib",
    }.get(platform.system(), "libclang.so")

    monkeypatch.setattr(
        _paths, "get_library_path", lambda shortname: b"libclang.so"
    )

    assert cindex.Config().get_filename() == expected


def test_get_filename_respects_explicit_library_file(
    monkeypatch, clear_config
):
    monkeypatch.setattr(cindex.Config, "library_file", "/custom/libclang.so")
    # Resolver would return something else; explicit override must win.
    monkeypatch.setattr(
        _paths, "get_library_path", lambda shortname: b"/other/libclang.so"
    )

    assert cindex.Config().get_filename() == "/custom/libclang.so"
