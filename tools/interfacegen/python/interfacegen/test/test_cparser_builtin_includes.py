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

"""Unit tests for the fallback builtin headers and the fatal-diagnostic gate.

``stddef.h`` and its siblings come from the compiler, not from the C library,
and the PyPI ``libclang`` wheel ships no resource directory to hold them. What
makes that worth a test rather than a note is the failure mode: clang reports
the miss and then recovers, so ``size_t`` arrives as ``int`` and every consumer
downstream -- the renderer, the wrapper dispatch, the emitted ``.pxd`` -- agrees
on a width the header never declared.

So there are two guarantees here. The headers resolve without a toolchain, and
where they cannot cover something the parse fails instead of quietly answering
with a different type.
"""

import textwrap

import clang.cindex
import pytest
from interfacegen.cparser import BUILTIN_INCLUDE_DIR, CParser

NO_TOOLCHAIN_TARGET = "--target=x86_64-unknown-linux-gnu"
"""Target for the tests that assert what clang does with a header it cannot find.

Whether a header can be found is a property of the target rather than of
the suite. Clang defaults to ``x86_64-pc-windows-msvc`` on a Windows host,
where it finds the UCRT's ``stddef.h`` by detecting the Visual Studio
installation, and predeclares ``size_t`` in MS-compatibility mode -- so
there is no miss left to assert. Neither ``-nostdsysteminc`` nor an empty
``INCLUDE`` suppresses that detection; naming a target with no toolchain to
find does, and follows what the data-model test below already does.
"""


def _parse(header_text: str, append_cflags=None):
    parser = CParser(
        "input.h",
        append_cflags=append_cflags or [],
        unsaved_files=[("input.h", textwrap.dedent(header_text))],
    )
    parser.parse()
    return parser


def _parm_types(parser, fn_name="probe"):
    """``{parameter name: pointee spelling}`` for one function."""
    for cursor in parser.cursor.get_children():
        if cursor.spelling == fn_name:
            return {
                arg.spelling: arg.type.get_pointee().spelling
                for arg in cursor.get_arguments()
            }
    raise KeyError(fn_name)


def _enum_values(parser):
    return {
        c.spelling: c.enum_value
        for c in parser.cursor.walk_preorder()
        if c.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL
    }


# ---------------------------------------------------------------------------
# the headers resolve
# ---------------------------------------------------------------------------


def test_compiler_supplied_typedefs_survive_without_a_toolchain():
    """The regression: these came back as ``int``.

    ``uint64_t`` is here as the control. It resolved even while ``size_t`` did
    not, because a glibc host supplies ``stdint.h`` and no host supplies
    ``stddef.h`` -- which is what made the failure look like a canonicalisation
    bug rather than a missing include.
    """
    parser = _parse(
        """\
        #include <stddef.h>
        #include <stdint.h>
        void probe(size_t *a, ptrdiff_t *b, uint64_t *c, uintptr_t *d);
        """
    )
    assert _parm_types(parser) == {
        "a": "size_t",
        "b": "ptrdiff_t",
        "c": "uint64_t",
        "d": "uintptr_t",
    }


def test_stdbool_resolves():
    parser = _parse(
        """\
        #include <stdbool.h>
        void probe(bool *a);
        """
    )
    assert _parm_types(parser) == {"a": "_Bool"}


def test_the_widths_follow_the_parse_target_not_the_host():
    """A declaration parsed for Windows must be sized for Windows.

    The headers name clang predefines rather than concrete C types for this
    reason: ``long`` is 64 bits under LP64 and 32 under LLP64, so a fixed
    spelling would hand Windows a binding sized for the machine that generated
    it.
    """
    src = """\
        #include <stddef.h>
        #include <stdint.h>
        void probe(size_t *a, int64_t *b);
        """
    lp64 = _parse(src, ["--target=x86_64-unknown-linux-gnu"])
    llp64 = _parse(src, ["--target=x86_64-pc-windows-msvc"])

    def canonical(parser, name):
        for cursor in parser.cursor.get_children():
            if cursor.spelling == "probe":
                for arg in cursor.get_arguments():
                    if arg.spelling == name:
                        return arg.type.get_pointee().get_canonical().spelling
        raise KeyError(name)

    assert canonical(lp64, "a") == "unsigned long"
    assert canonical(llp64, "a") == "unsigned long long"
    assert canonical(lp64, "b") == "long"
    assert canonical(llp64, "b") == "long long"


# ---------------------------------------------------------------------------
# the shim shadows the platform's stdint.h, so it has to be as complete
# ---------------------------------------------------------------------------


def test_limit_macros_are_usable_in_constant_expressions():
    parser = _parse(
        """\
        #include <stdint.h>
        enum probe_limits {
            p_int32_max = INT32_MAX,
            p_int8_min = INT8_MIN,
            p_uint16_max = UINT16_MAX,
            p_int_least16_max = INT_LEAST16_MAX,
            p_int_fast8_min = INT_FAST8_MIN,
            p_sig_atomic_max = SIG_ATOMIC_MAX,
        };
        """
    )
    values = _enum_values(parser)
    assert values["p_int32_max"] == 2147483647
    assert values["p_int8_min"] == -128
    assert values["p_uint16_max"] == 65535
    assert values["p_int_least16_max"] == 32767
    assert values["p_int_fast8_min"] == -128
    assert values["p_sig_atomic_max"] == 2147483647


def test_the_wide_limits_and_the_constant_constructors_are_right():
    """Checked by comparison, since these do not fit in an enum constant."""
    parser = _parse(
        """\
        #include <stddef.h>
        #include <stdint.h>
        enum probe_wide {
            p_uint64 = UINT64_MAX == (uint64_t)-1,
            p_size = SIZE_MAX == (size_t)-1,
            p_uintptr = UINTPTR_MAX == (uintptr_t)-1,
            p_ptrdiff = PTRDIFF_MAX == (ptrdiff_t)((~(uintptr_t)0) >> 1),
            p_int64_c = INT64_C(1) == (int64_t)1,
            p_uint64_c = UINT64_C(1) == (uint64_t)1,
            p_uint8_c = UINT8_C(255) == 255,
        };
        """
    )
    assert all(v == 1 for v in _enum_values(parser).values())


# ---------------------------------------------------------------------------
# what the fallback does not cover has to fail loudly
# ---------------------------------------------------------------------------


def test_a_header_that_cannot_be_opened_raises():
    with pytest.raises(
        clang.cindex.TranslationUnitLoadError, match="no_such_header.h"
    ):
        _parse(
            """\
            #include <no_such_header.h>
            void probe(missing_t *a);
            """
        )


def test_the_escape_hatch_permits_the_recovered_ast(monkeypatch):
    """Kept usable for a broken environment, and it shows what the gate buys:
    the parse that succeeds here answers ``int`` for a type the header never
    declared."""
    monkeypatch.setenv("INTERFACEGEN_ALLOW_FATAL_DIAGNOSTICS", "1")
    parser = _parse(
        """\
        #include <no_such_header.h>
        void probe(size_t *a);
        """,
        [NO_TOOLCHAIN_TARGET],
    )
    assert [
        d
        for d in parser.translation_unit.diagnostics
        if d.severity >= clang.cindex.Diagnostic.Fatal
    ], "the hatch is only interesting for a parse the gate would have rejected"
    assert _parm_types(parser) == {"a": "int"}


def test_deliberately_malformed_declarations_still_parse():
    """The gate is set at ``Fatal`` rather than ``Error`` because the renderer
    tests feed clang input it is expected to recover from, and real headers
    such as ``hiprand.h`` report errors when parsed in isolation while still
    yielding usable declarations."""
    parser = _parse(
        """\
        struct S { int[8] _x; };
        """
    )
    assert [
        d
        for d in parser.translation_unit.diagnostics
        if d.severity == clang.cindex.Diagnostic.Error
    ]


# ---------------------------------------------------------------------------
# a real toolchain takes precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("joined", [False, True], ids=["separated", "joined"])
def test_an_explicit_resource_dir_suppresses_the_fallback(tmp_path, joined):
    """A caller with a toolchain gets it, untouched.

    Every production path passes ``-resource-dir``; pointing it at an empty
    directory is how this asserts the fallback stood aside, since the headers
    then resolve from nowhere.

    Both spellings clang accepts are checked. The fallback goes on as
    ``-isystem``, which is searched ahead of the named resource directory, so
    a spelling the check does not recognise does not merely add the fallback
    -- it lets the fallback win over the caller's own headers.
    """
    resource_dir = (
        [f"-resource-dir={tmp_path}"]
        if joined
        else ["-resource-dir", str(tmp_path)]
    )
    with pytest.raises(
        clang.cindex.TranslationUnitLoadError, match="stddef.h"
    ):
        _parse(
            """\
            #include <stddef.h>
            void probe(size_t *a);
            """,
            resource_dir + [NO_TOOLCHAIN_TARGET],
        )


def test_the_fallback_headers_ship_with_the_package():
    """They are data files, so packaging drops them unless declared."""
    import os

    for name in ("stddef.h", "stdint.h", "stdbool.h"):
        assert os.path.exists(os.path.join(BUILTIN_INCLUDE_DIR, name))


if __name__ == "__main__":
    pytest.main([__file__])
