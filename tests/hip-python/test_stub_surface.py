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

"""Guard the hand-maintained ``hip/__init__.pyi`` against drift.

The stub next to the shim is written by hand and nothing regenerates it, so
a name added to ``__init__.py`` and forgotten here silently reverts consumers
to ``Any``.

The stub is parsed rather than grepped: a substring search is satisfied by a
mention inside a docstring or a longer identifier, so it would pass even
after a name was renamed out from under it.

Checked against the *installed* wheel, including the PEP 561 marker that
makes any of it visible in the first place.
"""

__author__ = "Advanced Micro Devices, Inc."

import ast
import inspect
import os

import pytest

hip_pkg = pytest.importorskip("hip")

#: The guaranteed surface. Neither the modules nor `__version__` are caught
#: by the scan over `vars(hip)` below.
REQUIRED_NAMES = ["VERSION", "__version__", "hip", "hiprtc"]


def _package_dir():
    return os.path.dirname(hip_pkg.__file__)


def _declared_names(body):
    """Names bound by a `.pyi` body: defs, classes, imports, attributes."""
    names = set()
    for node in body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


@pytest.fixture(scope="module")
def stub_names():
    stub = os.path.join(_package_dir(), "__init__.pyi")
    assert os.path.exists(stub), "hip ships no __init__.pyi"
    with open(stub, encoding="utf-8") as handle:
        return _declared_names(ast.parse(handle.read()).body)


def test_marker_is_shipped():
    # Without it every checker ignores the stub, which is the state this
    # package shipped in for its whole life.
    assert os.path.exists(os.path.join(_package_dir(), "py.typed"))


def test_stub_covers_the_eagerly_bound_names(stub_names):
    missing = sorted(
        name
        for name, value in vars(hip_pkg).items()
        if not name.startswith("_")
        # Imported modules (`importlib`) are implementation detail; the
        # binding modules are covered by REQUIRED_NAMES below.
        and not inspect.ismodule(value) and name not in stub_names
    )
    assert not missing, f"__init__.pyi does not declare {missing}"


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_stub_covers_the_guaranteed_names(name, stub_names):
    assert name in stub_names


def test_stub_keeps_the_fallback_for_the_optional_bindings(stub_names):
    # hipblas and friends live in wheels a user may not have installed, so
    # the stub cannot import them; `__getattr__` is what keeps them typed
    # as Any instead of erroring.
    assert "__getattr__" in stub_names
