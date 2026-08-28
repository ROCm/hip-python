# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

The env-var tier is Unix-only, so the tests covering it are too — see
``unix_only`` below.
"""

import sys
import types

import pytest
from rocm.bindings.util import paths


def _evict_rocm_sdk_modules(monkeypatch):
    """Remove cached ``rocm_sdk`` submodules so a subsequent block is effective."""
    for name in list(sys.modules):
        if name == "rocm_sdk" or name.startswith("rocm_sdk."):
            monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.fixture
def block_rocm_sdk(monkeypatch):
    """Make ``import rocm_sdk`` (and submodules) fail, forcing the env tier."""
    _evict_rocm_sdk_modules(monkeypatch)
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


# get_library_path skips the ROCM_PATH / ROCM_HOME tier on Windows: ROCm's DLLs
# are located through PATH there rather than under a <rocm>/lib directory, and
# returning an absolute path would be actively worse, because LoadLibrary does
# not search a full-path-loaded DLL's own directory for its dependencies — the
# rest of the ROCm stack it pulls in would go unfound. So on Windows these four
# resolutions land on the bare-name fallback ('clang.dll', 'LLVM.dll') instead of
# the fixture trees, and the expectations below describe Unix only.
unix_only = pytest.mark.skipif(
    sys.platform in ("win32", "cygwin"),
    reason="get_library_path's ROCM_PATH/ROCM_HOME tier is Unix-only",
)


@unix_only
def test_get_library_path_clang_rocm_home(
    tmp_path, monkeypatch, block_rocm_sdk
):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libclang.so").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path(
        "clang", bundled_location=_no_bundled(tmp_path)
    ) == str(llvm_lib / "libclang.so").encode("utf-8")


@unix_only
def test_get_library_path_clang_versioned_soname(
    tmp_path, monkeypatch, block_rocm_sdk
):
    llvm_lib = _make_llvm_lib(tmp_path)
    # Only the versioned soname is present (no bare libclang.so), as shipped by
    # rocm-sdk-core.
    (llvm_lib / "libclang.so.23.0git").touch()
    monkeypatch.setenv("ROCM_HOME", str(tmp_path))

    assert paths.get_library_path(
        "clang", bundled_location=_no_bundled(tmp_path)
    ).decode("utf-8") == str(llvm_lib / "libclang.so.23.0git")


@unix_only
def test_get_library_path_llvm_rocm_path(
    tmp_path, monkeypatch, block_rocm_sdk
):
    llvm_lib = _make_llvm_lib(tmp_path)
    (llvm_lib / "libLLVM.so.23.0git").touch()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))

    assert paths.get_library_path(
        "LLVM", bundled_location=_no_bundled(tmp_path)
    ).decode("utf-8") == str(llvm_lib / "libLLVM.so.23.0git")


@unix_only
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
    _evict_rocm_sdk_modules(monkeypatch)
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
    result = paths.get_library_path(
        "amdhip64", bundled_location=tmp_path / "empty"
    )

    assert result == str(lib_file).encode("utf-8")


def test_get_library_path_rocm_sdk_returns_str(tmp_path, monkeypatch):
    """The rocm_sdk tier also accepts plain ``str`` paths (forward-compat)."""
    _evict_rocm_sdk_modules(monkeypatch)
    lib_file = tmp_path / "libamdhip64.so"
    lib_file.touch()

    fake_rocm_sdk = types.ModuleType("rocm_sdk")
    fake_rocm_sdk.find_libraries = lambda shortname: [str(lib_file)]
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_rocm_sdk)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)

    result = paths.get_library_path(
        "amdhip64", bundled_location=tmp_path / "empty"
    )

    assert result == str(lib_file).encode("utf-8")


def _windows_or_unix_name(shortname, major=7, minor=14):
    """The filename a ROCm install uses for ``shortname`` on this platform."""
    if sys.platform in ("win32", "cygwin"):
        return f"{shortname}{major:02d}{minor:02d}.dll"
    return f"lib{shortname}.so"


def test_get_library_path_anchored_shortname(tmp_path, monkeypatch):
    """A library rocm_sdk does not register resolves beside one that it does.

    ``find_libraries`` raises for anything outside its registry, which is where
    hiprtc-builtins sits, so the only way a wheel install can name it is via a
    registered neighbour -- here hipRTC, installed in the same directory.
    """
    _evict_rocm_sdk_modules(monkeypatch)
    lib_dir = tmp_path / "_rocm_sdk_core" / "bin"
    lib_dir.mkdir(parents=True)
    anchor = lib_dir / _windows_or_unix_name("hiprtc")
    anchor.touch()
    builtins = lib_dir / _windows_or_unix_name("hiprtc-builtins")
    builtins.touch()

    def find_libraries(shortname):
        if shortname == "hiprtc":
            return [anchor]
        raise ModuleNotFoundError(shortname)  # what rocm_sdk raises

    fake_rocm_sdk = types.ModuleType("rocm_sdk")
    fake_rocm_sdk.find_libraries = find_libraries
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_rocm_sdk)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.delenv("ROCM_HOME", raising=False)

    result = paths.get_library_path(
        "hiprtc-builtins", bundled_location=tmp_path / "empty"
    )

    assert result == str(builtins).encode("utf-8")


def test_get_library_path_anchored_shortname_without_anchor(
    tmp_path, monkeypatch
):
    """No anchor means no resolution, not an exception.

    An install with no hipRTC at all -- a Windows machine carrying only the GPU
    driver -- has to reach the bare-name fallback so the caller can report the
    library as missing rather than crash inside path resolution.
    """
    _evict_rocm_sdk_modules(monkeypatch)
    empty = tmp_path / "empty"

    fake_rocm_sdk = types.ModuleType("rocm_sdk")
    fake_rocm_sdk.find_libraries = lambda shortname: []
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake_rocm_sdk)
    # Unsetting these does not empty the Unix env tier: it falls back to
    # /opt/rocm, so on a host with a traditional ROCm install the anchor
    # resolves there and this never reaches the fallback it is testing.
    # Naming a directory that holds no library is what makes the tier miss
    # wherever the suite runs.
    monkeypatch.setenv("ROCM_PATH", str(empty))
    monkeypatch.setenv("ROCM_HOME", str(empty))
    monkeypatch.setenv("PATH", str(empty))

    result = paths.get_library_path(
        "hiprtc-builtins", bundled_location=empty
    ).decode("utf-8")

    assert result == _bare_name("hiprtc-builtins")


def _bare_name(shortname):
    if sys.platform in ("win32", "cygwin"):
        return f"{shortname}.dll"
    if sys.platform == "darwin":
        return f"lib{shortname}.dylib"
    return f"lib{shortname}.so"


def _no_rocm_tree(tmp_path, monkeypatch):
    """Point every environment tier at a directory that holds no ROCm."""
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("ROCM_PATH", str(empty))
    monkeypatch.setenv("ROCM_HOME", str(empty))


def test_get_clang_resource_dir_beside_libclang(tmp_path, monkeypatch):
    """The usual case: the resource directory sits next to libclang."""
    _no_rocm_tree(tmp_path, monkeypatch)
    lib = tmp_path / "llvm" / "lib"
    resource_dir = lib / "clang" / "23"
    resource_dir.mkdir(parents=True)
    libclang = lib / _bare_name("clang")
    libclang.touch()

    assert paths.get_clang_resource_dir(str(libclang)) == str(resource_dir)


def test_get_clang_resource_dir_libclang_in_bin(tmp_path, monkeypatch):
    """Windows layout: libclang in bin, its resource directory in lib."""
    _no_rocm_tree(tmp_path, monkeypatch)
    llvm = tmp_path / "llvm"
    resource_dir = llvm / "lib" / "clang" / "23"
    resource_dir.mkdir(parents=True)
    (llvm / "bin").mkdir()
    libclang = llvm / "bin" / _bare_name("clang")
    libclang.touch()

    assert paths.get_clang_resource_dir(str(libclang)) == str(resource_dir)


def test_get_clang_resource_dir_highest_version(tmp_path, monkeypatch):
    """Several clang versions side by side resolve to the newest."""
    _no_rocm_tree(tmp_path, monkeypatch)
    lib = tmp_path / "llvm" / "lib"
    for version in ("21", "23"):
        (lib / "clang" / version).mkdir(parents=True)
    libclang = lib / _bare_name("clang")
    libclang.touch()

    assert paths.get_clang_resource_dir(str(libclang)) == str(
        lib / "clang" / "23"
    )


def test_get_clang_resource_dir_from_rocm_path(
    tmp_path, monkeypatch, block_rocm_sdk
):
    """Without a libclang to anchor on, the ROCm tree is searched.

    Both the traditional llvm/lib and the wheel/TheRock lib/llvm/lib layouts
    are covered; here the latter, which is the only one Windows ships.
    """
    resource_dir = tmp_path / "lib" / "llvm" / "lib" / "clang" / "23"
    resource_dir.mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.delenv("ROCM_HOME", raising=False)

    assert paths.get_clang_resource_dir("no_such_libclang") == str(
        resource_dir
    )


def test_get_clang_resource_dir_not_found(
    tmp_path, monkeypatch, block_rocm_sdk
):
    """Nothing anywhere is reported as such rather than guessed at."""
    _no_rocm_tree(tmp_path, monkeypatch)

    assert paths.get_clang_resource_dir("no_such_libclang") is None
