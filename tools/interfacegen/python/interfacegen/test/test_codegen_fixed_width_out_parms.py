# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression tests pinning fixed-width integer spellings in emitted modules.

Clang's canonical spelling for the stdint typedefs depends on the data model of
the host the generator runs on: ``uint64_t`` canonicalizes to ``unsigned long``
on LP64 Linux but to ``unsigned long long`` on LLP64 Windows, where
``unsigned long`` is only 32 bits wide. A generated tree is produced once and
compiled on every platform, so a canonicalized spelling is a defect twice over:

  * it truncates, when a 64-bit value passes through a 32-bit declaration or the
    callee writes 8 bytes into the 4-byte slot behind an out-parameter; and
  * it fails the build, because the ``.pxd`` declarations and the ``.pyx``
    bodies are rendered by separate code paths -- when only one of them
    canonicalizes, Cython rejects the mismatch outright with
    ``Cannot assign type 'unsigned long long *' to 'size_t *'``.

The typedef name is therefore what must reach both files. See
``typerender.fixed_width_typedef``.
"""

import textwrap

import pytest
from interfacegen.test._codegen_helpers import make_generator, write_module

#: Canonical spellings clang would substitute for the typedefs below. Their
#: absence is the property under test: whichever emitter handles the parameter,
#: none of them may resolve a fixed-width typedef down to a platform type.
_CANONICAL_LEAKS = ("unsigned long", "long long", "unsigned int")


def _emit(ctype, tmp_path):
    src = textwrap.dedent(
        f"""\
        #include <stdint.h>
        #include <stddef.h>
        /**
         * @brief Writes a value.
         * @param[out] out the value is written here.
         */
        int my_fn({ctype} *out);
        """
    )
    gen = make_generator(
        src, module_name="mod", runtime_linking=True, dll="libmy.so"
    )
    return write_module(gen, tmp_path)


def _emit_in_parm(ctype, tmp_path):
    src = textwrap.dedent(
        f"""\
        #include <stdint.h>
        #include <stddef.h>
        /**
         * @brief Takes a value.
         * @param[in] value the value.
         */
        int my_fn({ctype} value);
        """
    )
    gen = make_generator(
        src, module_name="mod", runtime_linking=True, dll="libmy.so"
    )
    return write_module(gen, tmp_path)


def _my_fn_region(text):
    """The emitted text for ``my_fn`` only.

    The modules also carry the predefined-macro dump, which mentions plenty of
    platform types unrelated to the parameter being checked.
    """
    start = text.find("my_fn")
    assert start != -1, f"no my_fn in emitted text:\n{text[:400]}"
    end = text.find("__all__", start)
    return text[start : end if end != -1 else len(text)]


@pytest.mark.parametrize(
    "ctype",
    ["size_t", "uint64_t", "uint32_t", "int64_t", "uintptr_t", "ptrdiff_t"],
)
def test_declaration_keeps_the_typedef(ctype, tmp_path):
    """The c-interface ``.pxd`` must declare the parameter with the typedef."""
    cypxd = _my_fn_region(_emit(ctype, tmp_path)["cymod.pxd"])
    assert f"{ctype} *" in cypxd, f"expected '{ctype} *' in:\n{cypxd}"


@pytest.mark.parametrize(
    "ctype",
    ["size_t", "uint64_t", "uint32_t", "int64_t", "uintptr_t", "ptrdiff_t"],
)
def test_body_agrees_with_the_declaration(ctype, tmp_path):
    """The ``.pyx`` body must name the same type it passes to the declaration,
    whether it does so through a cast or through a local it takes the address
    of. A canonical spelling on either side is what Cython rejects."""
    pyx = _my_fn_region(_emit(ctype, tmp_path)["mod.pyx"])
    assert ctype in pyx, f"expected '{ctype}' in:\n{pyx}"
    leaked = [c for c in _CANONICAL_LEAKS if c in pyx]
    assert not leaked, f"canonical spelling(s) {leaked} leaked into:\n{pyx}"


@pytest.mark.parametrize(
    "ctype", ["size_t", "uint64_t", "int64_t", "ptrdiff_t"]
)
def test_in_parm_documents_the_python_type_it_accepts(ctype, tmp_path):
    """The docstring names the Python types Cython converts from, and an
    integer typedef accepts ``int``.

    Keeping the typedef name means the autoconversion table sees ``size_t``
    rather than ``unsigned long``. A table that only lists the plain C integers
    falls through to its catch-all, which is how the sequence type ``list``
    ends up documented for a scalar -- one character at a time, since the
    catch-all returns a bare string where every other branch returns a tuple.
    """
    doc = _my_fn_region(_emit_in_parm(ctype, tmp_path)["mod.pyx"])
    assert "value (:py:obj:`~.int`)" in doc, f"expected int in:\n{doc}"


def test_plain_integer_spelling_is_not_rewritten(tmp_path):
    """Only the typedefs are substituted. A header that genuinely says
    ``unsigned long`` means the platform's long, and must keep saying so."""
    files = _emit("unsigned long", tmp_path)
    assert "unsigned long *" in _my_fn_region(files["cymod.pxd"])
    assert "unsigned long" in _my_fn_region(files["mod.pyx"])


if __name__ == "__main__":
    pytest.main([__file__])
