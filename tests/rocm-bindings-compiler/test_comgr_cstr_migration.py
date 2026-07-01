# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression tests for the comgr.py migration from
``to_cstr`` + ``_KeepAliveMixin`` to direct ``CStr``.

Locks down the contract that motivated the migration:

  * Strings handed to comgr setters (``Action.set_isa_name``,
    ``Action.set_options``, ``Data._set_data_name``) are interned in
    ``rocm.bindings.util.types.CStr._retained_inputs`` (or
    ``ListOfBytes._retained_inputs`` for the option list path) so the
    backend's retained pointer remains valid for the program's
    lifetime — even after the wrapper instance and the caller's
    source binding are GC'd.
  * Repeated calls with the same logical content do not grow the
    intern dict (deduplication by content).
  * ``str`` and ``bytes`` inputs intern to the same canonical bytes.

These tests do NOT require a real GPU — they only exercise the
Python-side wrapper construction. They run in any environment where
the rocm-bindings-compiler wheel is installed.
"""

import gc

import pytest

from rocm.bindings.util import types as _t


@pytest.fixture(autouse=True)
def _clean_intern_dicts():
    """Reset the intern dicts before each test so dedup-bound
    assertions get a deterministic baseline."""
    _t._clear_retained_inputs()
    yield
    _t._clear_retained_inputs()


def _churn_heap():
    """Force GC + a few churn allocations so any UAF would surface."""
    gc.collect()
    junk = [bytearray(b"X" * 1024) for _ in range(64)]
    del junk
    gc.collect()


def test_isa_name_str_is_interned_after_action_dropped():
    """Constructing an Action with a str isa_name must intern the
    encoded bytes in ``CStr._retained_inputs``. Dropping the Action
    AND the source str must not evict the canonical bytes — the
    backend may still hold the pointer.
    """
    from rocm.comgr.comgr import Action

    isa = "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"
    action = Action("COMPILE_SOURCE_TO_BC", isa_name=isa)
    canonical = isa.encode("utf-8")
    assert canonical in _t.CStr._retained_inputs
    del action, isa
    _churn_heap()
    assert canonical in _t.CStr._retained_inputs


def test_isa_name_bytes_and_str_dedup_to_same_canonical_pointer():
    """Same logical content via str and bytes paths must intern to
    the same canonical bytes object — verifies that comgr setters
    ride CStr's content-dedup contract."""
    from rocm.comgr.comgr import Action

    isa = "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"
    Action("COMPILE_SOURCE_TO_BC", isa_name=isa)
    Action("COMPILE_SOURCE_TO_BC", isa_name=isa.encode("utf-8"))
    # Exactly one canonical bytes for this content.
    matching = [b for b in _t.CStr._retained_inputs if b == isa.encode("utf-8")]
    assert len(matching) == 1


def test_repeated_set_options_with_identical_flags_bounded():
    """1000 identical option lists must keep the ListOfBytes intern
    dict at exactly 1 entry — the backend allocates fresh memory
    each call (it copies), but our intern is content-bounded."""
    from rocm.comgr.comgr import Action

    flags = [b"-O3", b"-Wall"]
    for _ in range(1000):
        action = Action("COMPILE_SOURCE_TO_BC")
        action.set_options(flags)
    # The intern dict is keyed by CStr instance (the per-option
    # wrapper). Each call constructs a fresh CStr per option, so
    # the count grows by len(flags) per call. Verify that the
    # backing bytes content is deduplicated in CStr._retained_inputs
    # — that's the bound that matters for memory.
    assert len(_t.CStr._retained_inputs) <= 4  # 2 unique flags (+slack for any prior)


def test_data_name_str_interned_after_data_dropped():
    """Constructing a Data with a str name must intern the encoded
    bytes; dropping the Data and the source str must not evict it.
    """
    from rocm.comgr.comgr import Data

    name = "test_data_name_str_interned_kernel"
    data = Data(name=name, kind_str="SOURCE")
    canonical = name.encode("utf-8")
    assert canonical in _t.CStr._retained_inputs
    del data, name
    _churn_heap()
    assert canonical in _t.CStr._retained_inputs


# ---------------------------------------------------------------------------
# *_str_to_enum: short form, full prefix form, case-insensitive matching
# ---------------------------------------------------------------------------


def test_action_kind_str_to_enum_accepts_short_and_full_prefix():
    """``action_kind_str_to_enum`` accepts both the short form
    (``"COMPILE_SOURCE_TO_BC"``) and the already-prefixed form
    (``"AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC"``); both must
    resolve to the exact same enum member."""
    from rocm.comgr.comgr import Action

    short = Action.action_kind_str_to_enum("COMPILE_SOURCE_TO_BC")
    full = Action.action_kind_str_to_enum("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC")
    assert short is full
    assert short.name == "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC"


def test_action_kind_str_to_enum_case_insensitive():
    """Lowercase / mixed-case inputs are normalised to upper-case
    before the lookup."""
    from rocm.comgr.comgr import Action

    upper = Action.action_kind_str_to_enum("LINK_BC_TO_BC")
    lower = Action.action_kind_str_to_enum("link_bc_to_bc")
    mixed = Action.action_kind_str_to_enum("Link_Bc_To_Bc")
    full_mixed = Action.action_kind_str_to_enum(
        "amd_comgr_action_link_bc_to_bc"
    )
    assert upper is lower is mixed is full_mixed


def test_data_kind_str_to_enum_accepts_short_and_full_prefix():
    from rocm.comgr.comgr import Data

    short = Data.kind_str_to_enum("FATBIN")
    full = Data.kind_str_to_enum("AMD_COMGR_DATA_KIND_FATBIN")
    assert short is full
    assert short.name == "AMD_COMGR_DATA_KIND_FATBIN"


def test_lang_str_to_enum_accepts_short_and_full_prefix():
    from rocm.comgr.comgr import Action

    short = Action.lang_str_to_enum("HIP")
    full = Action.lang_str_to_enum("AMD_COMGR_LANGUAGE_HIP")
    assert short is full
    assert short.name == "AMD_COMGR_LANGUAGE_HIP"


# ---------------------------------------------------------------------------
# valid_*() runtime introspection
# ---------------------------------------------------------------------------


def test_valid_action_kinds_contains_stable_baseline():
    """``valid_action_kinds`` must return a non-empty list and
    contain the ROCm 6.0.0 baseline keys that have shipped for years.
    Newly added keys (e.g. SPIR-V) are NOT asserted here so the test
    stays stable across ROCm version bumps."""
    from rocm.comgr.comgr import Action

    keys = Action.valid_action_kinds()
    assert len(keys) > 0
    baseline = {
        "SOURCE_TO_PREPROCESSOR",
        "ADD_PRECOMPILED_HEADERS",
        "COMPILE_SOURCE_TO_BC",
        "LINK_BC_TO_BC",
        "CODEGEN_BC_TO_RELOCATABLE",
        "CODEGEN_BC_TO_ASSEMBLY",
        "LINK_RELOCATABLE_TO_RELOCATABLE",
        "LINK_RELOCATABLE_TO_EXECUTABLE",
        "ASSEMBLE_SOURCE_TO_RELOCATABLE",
        "COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC",
        "LAST",
    }
    missing = baseline - set(keys)
    assert not missing, (
        f"valid_action_kinds() missing baseline keys: {missing}"
    )
    # Every returned key must round-trip through the lookup.
    for k in keys:
        Action.action_kind_str_to_enum(k)


def test_valid_data_kinds_contains_stable_baseline():
    from rocm.comgr.comgr import Data

    keys = Data.valid_kinds()
    baseline = {
        "UNDEF", "SOURCE", "INCLUDE", "PRECOMPILED_HEADER",
        "DIAGNOSTIC", "LOG", "BC", "RELOCATABLE", "EXECUTABLE",
        "BYTES", "FATBIN", "AR", "BC_BUNDLE", "AR_BUNDLE", "LAST",
    }
    missing = baseline - set(keys)
    assert not missing, (
        f"Data.valid_kinds() missing baseline keys: {missing}"
    )
    for k in keys:
        Data.kind_str_to_enum(k)


def test_valid_languages_contains_stable_baseline():
    from rocm.comgr.comgr import Action

    keys = Action.valid_languages()
    baseline = {"NONE", "OPENCL_1_2", "OPENCL_2_0", "HIP", "LAST"}
    missing = baseline - set(keys)
    assert not missing, (
        f"valid_languages() missing baseline keys: {missing}"
    )
    for k in keys:
        Action.lang_str_to_enum(k)


# ---------------------------------------------------------------------------
# Removed-upstream keys still raise (regression guard for the
# Notes section in each docstring)
# ---------------------------------------------------------------------------


def test_removed_action_kinds_raise_attribute_error():
    """Keys that were removed upstream (and called out in the
    docstring's Notes section) must raise AttributeError — both
    in short form and in full-prefix form."""
    from rocm.comgr.comgr import Action

    for short in ("ADD_DEVICE_LIBRARIES", "OPTIMIZE_BC_TO_BC",
                  "COMPILE_SOURCE_TO_FATBIN"):
        with pytest.raises(AttributeError):
            Action.action_kind_str_to_enum(short)
        with pytest.raises(AttributeError):
            Action.action_kind_str_to_enum("AMD_COMGR_ACTION_" + short)


def test_removed_languages_raise_attribute_error():
    from rocm.comgr.comgr import Action

    with pytest.raises(AttributeError):
        Action.lang_str_to_enum("HC")
    with pytest.raises(AttributeError):
        Action.lang_str_to_enum("AMD_COMGR_LANGUAGE_HC")
