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

"""Regression tests for `_is_useless_macro` and the recipe `node_filter`s
that wire it in.

Two categories of macro should be dropped at codegen time when they
appear inside a header that the prefix-admit recipes (hsa, amdsmi,
hipdnn_backend, hipblaslt, hipsparselt, hiptensor) consume:

A. **Visibility / linkage / deprecation attribute macros** — bodies
   expand to ``__attribute__(...)`` decorators with no Python value
   (e.g. ``HSA_API_EXPORT``, ``AMDSMI_DEPRECATED``).
B. **Header guards** — the conventional ``#define <FILE>_H`` /
   ``<FILE>_H_`` at the top of a header. Body is empty or `1`.

The body-aware check is critical for category B: a legitimate
constant whose name happens to end in ``_H`` (e.g. some hypothetical
``FOO_H = 42``) must NOT be filtered.
"""

import pytest
from _codegen_helpers import build_root
from interfacegen.support.recipes.rocm import (
    _is_attribute_macro,
    _is_header_guard_macro,
    _is_useless_macro,
    hipblaslt,
    hipdnn_backend,
    hipsparselt,
    hiptensor,
)
from interfacegen.tree import MacroDefinition


def _macros(root):
    """Yield every `MacroDefinition` in the parsed tree."""
    for n in root.walk(postorder=False):
        if isinstance(n, MacroDefinition):
            yield n


def _macro_named(root, name):
    for m in _macros(root):
        if m.name == name:
            return m
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Category A — attribute macros
# ---------------------------------------------------------------------------


def test_is_attribute_macro_matches_visibility_decorator():
    root = build_root(
        '#define FOO_API_EXPORT __attribute__((visibility("default")))\n'
    )
    m = _macro_named(root, "FOO_API_EXPORT")
    assert _is_attribute_macro(m) is True
    assert _is_useless_macro(m) is True


def test_is_attribute_macro_matches_deprecated():
    root = build_root("#define FOO_DEPRECATED __attribute__((deprecated))\n")
    m = _macro_named(root, "FOO_DEPRECATED")
    assert _is_attribute_macro(m) is True
    assert _is_useless_macro(m) is True


def test_is_attribute_macro_matches_no_export():
    root = build_root(
        '#define FOO_NO_EXPORT __attribute__((visibility("hidden")))\n'
    )
    m = _macro_named(root, "FOO_NO_EXPORT")
    assert _is_attribute_macro(m) is True


def test_is_attribute_macro_matches_export_decorator_and_api_call():
    """`*_EXPORT_DECORATOR`, `*_API`, `*_CALL` — all known suffixes
    that headers like hsa.h define for visibility / calling-convention
    decoration."""
    root = build_root(
        '#define FOO_EXPORT_DECORATOR __attribute__((visibility("default")))\n'
        "#define FOO_API\n"
        "#define FOO_CALL\n"
    )
    for name in ("FOO_EXPORT_DECORATOR", "FOO_API", "FOO_CALL"):
        m = _macro_named(root, name)
        assert _is_attribute_macro(m) is True, name


# ---------------------------------------------------------------------------
# Category B — header guards
# ---------------------------------------------------------------------------


def test_is_header_guard_macro_empty_body():
    """`#define FOO_H` (no body) is a textbook header guard."""
    root = build_root("#define FOO_H\n")
    m = _macro_named(root, "FOO_H")
    assert _is_header_guard_macro(m) is True
    assert _is_useless_macro(m) is True


def test_is_header_guard_macro_underscore_suffix():
    """`#define FOO_H_` (trailing underscore variant)."""
    root = build_root("#define FOO_RUNTIME_INC_H_\n")
    m = _macro_named(root, "FOO_RUNTIME_INC_H_")
    assert _is_header_guard_macro(m) is True


def test_is_header_guard_macro_body_one_kept_as_guard():
    """`#define FOO_H 1` is the alternate header-guard convention; the
    body is just the literal `1` token. Treat as guard."""
    root = build_root("#define FOO_H 1\n")
    m = _macro_named(root, "FOO_H")
    assert _is_header_guard_macro(m) is True


def test_is_header_guard_macro_body_with_value_is_kept():
    """A constant whose name happens to end in `_H` but that carries a
    real value (e.g. `42`, `0xff`) must NOT be filtered as a guard."""
    root = build_root("#define FOO_H 42\n")
    m = _macro_named(root, "FOO_H")
    assert _is_header_guard_macro(m) is False
    assert _is_useless_macro(m) is False


# ---------------------------------------------------------------------------
# Negative cases — normal int / string macros must NOT be flagged
# ---------------------------------------------------------------------------


def test_normal_int_macro_is_not_useless():
    root = build_root("#define HIPBLASLT_VERSION_MAJOR 1\n")
    m = _macro_named(root, "HIPBLASLT_VERSION_MAJOR")
    assert _is_attribute_macro(m) is False
    assert _is_header_guard_macro(m) is False
    assert _is_useless_macro(m) is False


def test_normal_size_macro_is_not_useless():
    root = build_root("#define AMDSMI_MAX_STRING_LENGTH 256\n")
    m = _macro_named(root, "AMDSMI_MAX_STRING_LENGTH")
    assert _is_useless_macro(m) is False


# ---------------------------------------------------------------------------
# Per-recipe `node_filter` integration: the early-reject in each
# prefix-admit recipe must drop attribute / guard macros even when
# they pass the name prefix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe,prefix",
    [
        (hipblaslt, "HIPBLASLT"),
        (hiptensor, "HIPTENSOR"),
        (hipdnn_backend, "HIPDNN"),
        (hipsparselt, "HIPSPARSELT"),
    ],
)
def test_recipe_node_filter_drops_useless_macros(recipe, prefix):
    """Synthesise an attribute macro + a header guard with the recipe's
    prefix and assert the recipe's `node_filter` rejects both, while
    accepting a normal int macro with the same prefix."""
    root = build_root(
        f'#define {prefix}_API_EXPORT __attribute__((visibility("default")))\n'
        f"#define {prefix}_H\n"
        f"#define {prefix}_VERSION_MAJOR 1\n"
    )
    attr_m = _macro_named(root, f"{prefix}_API_EXPORT")
    guard_m = _macro_named(root, f"{prefix}_H")
    int_m = _macro_named(root, f"{prefix}_VERSION_MAJOR")

    assert (
        recipe.node_filter(attr_m) is False
    ), f"{recipe.__name__}.node_filter must drop attribute macros"
    assert (
        recipe.node_filter(guard_m) is False
    ), f"{recipe.__name__}.node_filter must drop header-guard macros"
    assert (
        recipe.node_filter(int_m) is True
    ), f"{recipe.__name__}.node_filter must keep normal int macros"


# ---------------------------------------------------------------------------
# hipblaslt `_CODEGEN_BLOCKLIST` — C++-only macro bodies (static_cast /
# static_assert / bare git-hash token) must be dropped from export even
# though they carry the `HIPBLASLT_` prefix, because the default `int`
# macro_type would emit uncompilable `__Pyx_PyLong_From_int(MACRO)`.
# ---------------------------------------------------------------------------


def test_hipblaslt_node_filter_drops_cxx_only_macros():
    """Every name in `hipblaslt._CODEGEN_BLOCKLIST` is rejected by
    `hipblaslt.node_filter`, while a normal `HIPBLASLT_*` int macro is
    still accepted."""
    src = (
        "".join(f"#define {name} 0\n" for name in hipblaslt._CODEGEN_BLOCKLIST)
        + "#define HIPBLASLT_VERSION_MAJOR 1\n"
    )
    root = build_root(src)

    for name in hipblaslt._CODEGEN_BLOCKLIST:
        m = _macro_named(root, name)
        assert (
            hipblaslt.node_filter(m) is False
        ), f"hipblaslt.node_filter must drop blocklisted macro {name}"

    keep = _macro_named(root, "HIPBLASLT_VERSION_MAJOR")
    assert (
        hipblaslt.node_filter(keep) is True
    ), "hipblaslt.node_filter must keep normal HIPBLASLT_* int macros"
