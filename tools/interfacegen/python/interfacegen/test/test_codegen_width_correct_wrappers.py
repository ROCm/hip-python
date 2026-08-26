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

"""Regression tests for the util-type wrapper chosen for a pointer parameter.

The wrapper has to agree with the declaration about the element width, and the
declaration is generated once and compiled everywhere. Dispatching on the
innermost *canonical* clang kind cannot deliver that: clang resolves ``size_t``
to ``ULONG`` on LP64 Linux and to ``ULONGLONG`` on LLP64 Windows, so the same
header picks ``ListOfUnsignedLong`` on one host and falls through to the opaque
``Pointer`` on the other -- and ``unsigned long`` is 32 bits on Windows, so the
first choice allocates 4 bytes for a slot the callee writes 8 into.

The width a declaration pins is asked for instead (``typerender``'s
``fixed_width_typedef``, reached through ``tree.Typed.fixed_width_typedef``), so
every assertion below holds whatever host the generator ran on. Parameters that
genuinely say ``int`` / ``long`` keep the plain wrappers, which agree with their
declaration on both data models.

``ssize_t`` is deliberately absent: it is not declared on Windows, so it cannot
be part of a host-independent test. ``ptrdiff_t`` covers the same spec.
"""

import textwrap

import pytest
from interfacegen.test._codegen_helpers import make_generator, write_module

_UTIL_PREFIX = "rocm.bindings.util.types."


def _emit(ctype, tmp_path, *, rank, aliases=None, specs=None, prelude=""):
    src = textwrap.dedent(
        f"""\
        #include <stdint.h>
        #include <stddef.h>
        {prelude}
        int my_fn({ctype} *out);
        """
    )
    gen = make_generator(
        src,
        module_name="mod",
        runtime_linking=True,
        dll="libmy.so",
        ptr_rank=lambda node: rank,
        typedef_aliases=aliases,
        typedef_specs=specs,
    )
    return write_module(gen, tmp_path)


def _my_fn_region(text):
    """The emitted text for ``my_fn`` only; the modules also carry the
    predefined-macro dump, which mentions plenty of unrelated platform types.
    """
    start = text.find("my_fn")
    assert start != -1, f"no my_fn in emitted text:\n{text[:400]}"
    end = text.find("__all__", start)
    return text[start : end if end != -1 else len(text)]


def _assert_wrapper(files, expected):
    pyx = _my_fn_region(files["mod.pyx"])
    assert _UTIL_PREFIX + expected in pyx, f"expected '{expected}' in:\n{pyx}"


# ---------------------------------------------------------------------------
# the seven 64-bit typedefs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctype", ["size_t", "uint64_t", "uintptr_t"])
def test_unsigned_64_bit_buffer_is_listofuint64(ctype, tmp_path):
    _assert_wrapper(_emit(ctype, tmp_path, rank=1), "ListOfUInt64")


@pytest.mark.parametrize("ctype", ["int64_t", "ptrdiff_t", "intptr_t"])
def test_signed_64_bit_buffer_is_listofint64(ctype, tmp_path):
    _assert_wrapper(_emit(ctype, tmp_path, rank=1), "ListOfInt64")


@pytest.mark.parametrize("ctype", ["size_t", "uint64_t", "uintptr_t"])
def test_unsigned_64_bit_scalar_is_pointertouint64(ctype, tmp_path):
    _assert_wrapper(_emit(ctype, tmp_path, rank=0), "PointerToUInt64")


@pytest.mark.parametrize("ctype", ["int64_t", "ptrdiff_t", "intptr_t"])
def test_signed_64_bit_scalar_is_pointertoint64(ctype, tmp_path):
    _assert_wrapper(_emit(ctype, tmp_path, rank=0), "PointerToInt64")


# ---------------------------------------------------------------------------
# plain C integers keep the wrappers that match their declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctype,expected",
    [
        ("long", "ListOfLong"),
        ("unsigned long", "ListOfUnsignedLong"),
        ("int", "ListOfInt"),
        ("unsigned int", "ListOfUnsigned"),
        # 32-bit typedefs pin a width too, but no wrapper claims that spec:
        # they fall through to the canonical kinds, which are 32 bits on both
        # data models and therefore already correct.
        ("int32_t", "ListOfInt"),
        ("uint32_t", "ListOfUnsigned"),
    ],
)
def test_unpinned_and_32_bit_widths_keep_the_plain_wrappers(
    ctype, expected, tmp_path
):
    _assert_wrapper(_emit(ctype, tmp_path, rank=1), expected)


# ---------------------------------------------------------------------------
# a library typedef reaches the same wrapper through either per-module map
# ---------------------------------------------------------------------------

#: Stands in for hipFILE's ``hoff_t``: an alias whose definition differs per
#: platform, spelled here as the LP64 one.
_PRELUDE = "typedef long myoff_t;"


def test_alias_map_reaches_the_fixed_width_wrapper(tmp_path):
    files = _emit(
        "myoff_t",
        tmp_path,
        rank=0,
        aliases={"myoff_t": "int64_t"},
        prelude=_PRELUDE,
    )
    _assert_wrapper(files, "PointerToInt64")


def test_spec_map_pins_the_wrapper_and_the_spelling(tmp_path):
    """A width declared through ``typedef_specs`` decides the emitted spelling
    as well, because both come out of the same lookup. Were they resolved
    separately, an 8-byte wrapper could end up behind a declaration that still
    said ``long`` -- 4 bytes on Windows, the very mismatch this dispatch
    exists to remove."""
    files = _emit(
        "myoff_t",
        tmp_path,
        rank=0,
        specs={"myoff_t": (True, 64)},
        prelude=_PRELUDE,
    )
    _assert_wrapper(files, "PointerToInt64")
    cypxd = _my_fn_region(files["cymod.pxd"])
    assert "int64_t *" in cypxd, f"expected 'int64_t *' in:\n{cypxd}"
    assert "long *" not in cypxd, f"canonical spelling survived in:\n{cypxd}"


def test_without_a_map_the_library_typedef_keeps_the_host_wrapper(tmp_path):
    """Both maps are opt-in per module: an unlisted typedef pins no width, so
    it keeps resolving to the host's canonical kind."""
    files = _emit("myoff_t", tmp_path, rank=0, prelude=_PRELUDE)
    _assert_wrapper(files, "PointerToLong")


if __name__ == "__main__":
    pytest.main([__file__])
