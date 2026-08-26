# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Unit tests for ``tree.Typed.fixed_width_typedef`` and the per-module maps.

The accessor is the single place that answers what width a declaration pins.
The renderer takes the spelling it returns and the wrapper dispatch takes the
spec, so its contract is what keeps the emitted ``.pxd`` and the chosen
``ListOf*`` / ``PointerTo*`` in agreement -- see
``test_codegen_width_correct_wrappers.py`` for the end-to-end assertions.

Every statement here is host-independent, which the canonical clang kind these
consumers used to read could never be.
"""

import textwrap

import pytest
from interfacegen import cython
from interfacegen.test._codegen_helpers import make_generator


def _parm(decl, *, prelude="", aliases=None, specs=None, name="out"):
    src = textwrap.dedent(
        f"""\
        #include <stdint.h>
        #include <stddef.h>
        {prelude}
        int my_fn({decl});
        """
    )
    gen = make_generator(
        src,
        module_name="mod",
        runtime_linking=True,
        dll="libmy.so",
        typedef_aliases=aliases,
        typedef_specs=specs,
    )
    return next(
        n
        for n in gen.backend.root.walk(postorder=False)
        if isinstance(n, cython.Parm) and n.name == name
    )


# ---------------------------------------------------------------------------
# the built-in stdint vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctype,expected",
    [
        ("size_t", ("size_t", (False, 64))),
        ("uint64_t", ("uint64_t", (False, 64))),
        ("uintptr_t", ("uintptr_t", (False, 64))),
        ("int64_t", ("int64_t", (True, 64))),
        ("ptrdiff_t", ("ptrdiff_t", (True, 64))),
        ("intptr_t", ("intptr_t", (True, 64))),
        ("uint32_t", ("uint32_t", (False, 32))),
        ("int8_t", ("int8_t", (True, 8))),
    ],
)
def test_stdint_typedef_answers_its_own_name_and_spec(ctype, expected):
    assert _parm(f"{ctype} *out").fixed_width_typedef() == expected


def test_typedef_of_a_fixed_width_typedef_answers_the_inner_one():
    """An alias adds no width information, so the fixed-width name it wraps is
    the meaningful answer."""
    parm = _parm("my_size_t *out", prelude="typedef size_t my_size_t;")
    assert parm.fixed_width_typedef() == ("size_t", (False, 64))


@pytest.mark.parametrize(
    "ctype", ["long", "unsigned long", "int", "void", "char"]
)
def test_plain_c_types_pin_no_width(ctype):
    assert _parm(f"{ctype} *out").fixed_width_typedef() is None


def test_pointer_and_array_layers_are_skipped():
    """Only the leaf carries the width; the outer layers are rendered from the
    canonical walk by the caller."""
    assert _parm("size_t **out").fixed_width_typedef() == (
        "size_t",
        (False, 64),
    )
    assert _parm("const size_t *out").fixed_width_typedef() == (
        "size_t",
        (False, 64),
    )


# ---------------------------------------------------------------------------
# the two per-module maps
# ---------------------------------------------------------------------------

_PRELUDE = "typedef long myoff_t;"


def test_alias_map_answers_the_stdint_name_it_points_at():
    parm = _parm(
        "myoff_t *out", prelude=_PRELUDE, aliases={"myoff_t": "int64_t"}
    )
    assert parm.fixed_width_typedef() == ("int64_t", (True, 64))


def test_spec_map_answers_the_stdint_name_denoting_the_width():
    """A spec states the fact; the spelling to emit follows from it, so the two
    maps are two vocabularies for the same answer."""
    parm = _parm(
        "myoff_t *out", prelude=_PRELUDE, specs={"myoff_t": (True, 64)}
    )
    assert parm.fixed_width_typedef() == ("int64_t", (True, 64))


def test_unlisted_library_typedef_pins_no_width():
    assert (
        _parm("myoff_t *out", prelude=_PRELUDE).fixed_width_typedef() is None
    )


# ---------------------------------------------------------------------------
# construction-time validation
#
# An entry that silently resolves to nothing leaves the typedef canonicalized
# to the host's spelling -- the mismatch these maps exist to remove -- so a
# recipe typo has to fail here rather than in a shipped ``.pxd``.
# ---------------------------------------------------------------------------


def test_alias_to_an_unknown_typedef_is_rejected():
    with pytest.raises(ValueError, match="myoff_t"):
        _parm("myoff_t *out", prelude=_PRELUDE, aliases={"myoff_t": "off_t"})


def test_spec_with_an_unknown_width_is_rejected():
    with pytest.raises(ValueError, match="myoff_t"):
        _parm("myoff_t *out", prelude=_PRELUDE, specs={"myoff_t": (True, 42)})


def test_a_typedef_declared_in_both_maps_is_rejected():
    with pytest.raises(ValueError, match="both"):
        _parm(
            "myoff_t *out",
            prelude=_PRELUDE,
            aliases={"myoff_t": "int64_t"},
            specs={"myoff_t": (True, 64)},
        )


if __name__ == "__main__":
    pytest.main([__file__])
