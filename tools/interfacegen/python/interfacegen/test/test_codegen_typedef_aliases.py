# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression tests for the per-module ``typedef_aliases`` map.

``test_codegen_fixed_width_out_parms.py`` covers the typedefs of ``<stdint.h>``,
whose names the renderer keeps because Cython knows them. Libraries define their
own width-carrying aliases too -- hipFILE's ``hoff_t`` is ``off_t`` on POSIX and
``__int64`` on Windows -- and those canonicalize to whichever spelling the
codegen host uses: ``long`` on LP64, ``long long`` on LLP64. Nothing
hand-written can then name the parameter portably, because Cython compares
pointer types by identity and a single cast cannot be both.

``typedef_aliases`` maps such a name to the stdint typedef denoting the same C
type everywhere, so the emitted declaration reads the same wherever the
generator ran. See ``typerender.fixed_width_typedef``.
"""

import textwrap

import pytest
from interfacegen.test._codegen_helpers import make_generator, write_module

_HEADER = textwrap.dedent(
    """\
    #include <stdint.h>
    /* Stands in for hipFILE's hoff_t: an alias whose definition differs per
       platform, spelled here as the LP64 one. */
    typedef long myoff_t;
    /**
     * @brief Writes an offset.
     * @param[out] out the offset is written here.
     */
    int my_fn(myoff_t *out);
    """
)


def _emit(tmp_path, typedef_aliases=None):
    gen = make_generator(
        _HEADER,
        module_name="mod",
        runtime_linking=True,
        dll="libmy.so",
        typedef_aliases=typedef_aliases,
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


def test_alias_replaces_the_canonical_spelling(tmp_path):
    cypxd = _my_fn_region(_emit(tmp_path, {"myoff_t": "int64_t"})["cymod.pxd"])
    assert "int64_t *" in cypxd, f"expected 'int64_t *' in:\n{cypxd}"
    assert "long *" not in cypxd, f"canonical spelling survived in:\n{cypxd}"


def test_alias_reaches_the_body(tmp_path):
    """The ``.pyx`` body declares the local the callee writes to. It has to
    agree with the ``.pxd``, or Cython rejects the call outright."""
    pyx = _my_fn_region(_emit(tmp_path, {"myoff_t": "int64_t"})["mod.pyx"])
    assert "int64_t" in pyx, f"expected 'int64_t' in:\n{pyx}"


def test_without_the_map_the_typedef_is_canonicalized(tmp_path):
    """The mapping is opt-in per module: an unlisted typedef keeps resolving to
    the host's spelling, which is what every other library relies on."""
    cypxd = _my_fn_region(_emit(tmp_path)["cymod.pxd"])
    assert "long *" in cypxd, f"expected 'long *' in:\n{cypxd}"
    assert "int64_t *" not in cypxd


def test_unlisted_typedefs_are_untouched(tmp_path):
    """Only the named alias is substituted."""
    cypxd = _my_fn_region(_emit(tmp_path, {"other_t": "int64_t"})["cymod.pxd"])
    assert "long *" in cypxd, f"expected 'long *' in:\n{cypxd}"


if __name__ == "__main__":
    pytest.main([__file__])
