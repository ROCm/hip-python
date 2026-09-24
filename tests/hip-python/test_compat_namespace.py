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

"""What the ``hip`` backward-compatibility wheel promises.

``from hip import hip, hiprtc`` has to keep working for code written against
hip-python 3.x. The shim imports every binding it finds at import time, which
leaves four things to check: module identity, the ``hiprtc.ext`` helper, the
version metadata, and how it tells a binding whose wheel is not installed
from one that is installed and broken.

No GPU and no HIP runtime are needed here — nothing below calls into
``libamdhip64``. The suite runs against the *installed* wheel, so it is the
artifact users consume that is checked.
"""

__author__ = "Advanced Micro Devices, Inc."

import builtins
import importlib
import importlib.metadata
import importlib.util
import sys

import pytest

hip_pkg = pytest.importorskip("hip")

#: Names the shim re-exports from the generated `rocm.version`.
VERSION_METADATA = [
    "ROCM_VERSION",
    "ROCM_VERSION_NAME",
    "ROCM_VERSION_TUPLE",
    "rocm_version_name",
    "rocm_version_tuple",
    "HIP_VERSION",
    "HIP_VERSION_NAME",
    "HIP_VERSION_TUPLE",
    "hip_version_name",
    "hip_version_tuple",
]


def _installed_version():
    """The version the shim reports, or None where neither wheel is found."""
    for dist in ("hip-python", "rocm-bindings-hip"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


# ---------------------------------------------------------------------------
# The modules it re-exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hip", "hiprtc"])
def test_reexports_the_rocm_bindings_module(name):
    modern = importlib.import_module(f"rocm.bindings.{name}")
    assert getattr(hip_pkg, name) is modern


@pytest.mark.parametrize("name", ["hip", "hiprtc", "_util"])
def test_the_guaranteed_bindings_are_bound_at_import(name):
    # They come with the hard dependencies, so they are in dir() and in tab
    # completion rather than appearing on first access.
    assert name in vars(hip_pkg)


def test_the_legacy_import_spelling_works():
    # The one line every hip-python 3.x program starts with.
    from hip import hip, hiprtc

    assert hip is importlib.import_module("rocm.bindings.hip")
    assert hiprtc is importlib.import_module("rocm.bindings.hiprtc")


@pytest.mark.parametrize("name", ["hipblas", "rccl", "amd_comgr"])
def test_reexports_the_optional_bindings(name):
    modern = pytest.importorskip(f"rocm.bindings.{name}")
    assert getattr(hip_pkg, name) is modern
    assert name in vars(hip_pkg)


def test_hiprtc_carries_the_ext_helper():
    # `hiprtc.ext.hiprtcLinkCreate2` is how hip-python 3.x code built link
    # options. The helper is a sibling of rocm.bindings.hiprtc, not an
    # attribute of rocm.bindings.hip; attaching it from there failed silently.
    pyext = pytest.importorskip("rocm.bindings.hiprtc_pyext")
    assert hip_pkg.hiprtc.ext is pyext
    assert hasattr(hip_pkg.hiprtc.ext, "hiprtcLinkCreate2")


def test_the_private_util_package_keeps_its_old_spelling():
    # Downstream code, numba-hip among it, reached for `hip._util.types`.
    from rocm.bindings.util.types import Pointer

    assert hip_pkg._util is importlib.import_module("rocm.bindings.util")
    assert hip_pkg._util.types.Pointer is Pointer


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------


def test_version_matches_the_installed_distribution():
    expected = _installed_version()
    if expected is None:
        pytest.skip("neither hip-python nor rocm-bindings-hip is installed")
    assert hip_pkg.VERSION == expected
    assert hip_pkg.__version__ == expected


@pytest.mark.parametrize("name", VERSION_METADATA)
def test_version_metadata_is_the_object_from_rocm_version(name):
    rocm_version = pytest.importorskip("rocm.version")
    assert getattr(hip_pkg, name) is getattr(rocm_version, name)


# ---------------------------------------------------------------------------
# Bindings that are absent or broken
# ---------------------------------------------------------------------------


def test_unknown_name_raises_attribute_error():
    # ModuleNotFoundError here would make hasattr() raise instead of
    # answering False, which breaks inspect, copy and REPL completion.
    assert hasattr(hip_pkg, "not_a_binding") is False
    with pytest.raises(AttributeError, match="not_a_binding"):
        hip_pkg.not_a_binding


def test_an_uninstalled_binding_leaves_its_name_unbound(monkeypatch):
    """`pip install hip-python` alone has to import cleanly."""
    pytest.importorskip("rocm.bindings.hipblas")
    real_find_spec = importlib.util.find_spec

    def hide_hipblas(name, package=None):
        if name == "rocm.bindings.hipblas":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", hide_hipblas)
    reimported = _reimport_with(monkeypatch, _import_failing_on("hipblas"))

    assert not hasattr(reimported, "hipblas")
    assert hasattr(reimported, "hip")


def test_a_binding_that_fails_to_load_stays_loud(monkeypatch):
    """Swallowing this is how `hip.hiprtc.ext` went missing for two releases."""
    pytest.importorskip("rocm.bindings.hipblas")

    with pytest.raises(ImportError, match="libamdhip64_is_missing"):
        _reimport_with(monkeypatch, _import_failing_on("hipblas"))


def _import_failing_on(binding):
    """An `__import__` that refuses to hand out one rocm.bindings module."""
    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rocm.bindings" and binding in (fromlist or ()):
            raise ImportError(
                "libamdhip64_is_missing.so: cannot open shared object file",
                name=name,
            )
        return real_import(name, globals, locals, fromlist, level)

    return failing_import


def _reimport_with(monkeypatch, import_hook):
    """Import `hip` afresh under `import_hook`, then restore the real one."""
    monkeypatch.delitem(sys.modules, "hip")
    monkeypatch.setattr(builtins, "__import__", import_hook)
    return importlib.import_module("hip")
