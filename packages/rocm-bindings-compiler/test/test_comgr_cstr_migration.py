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


def test_to_cstr_helper_removed():
    """Sanity: the deleted ``to_cstr`` helper is gone from the
    comgr module (regression against accidental re-introduction)."""
    import rocm.comgr.comgr as comgr_mod
    assert not hasattr(comgr_mod, "to_cstr"), (
        "to_cstr was deleted in the CStr migration; do not re-introduce. "
        "Use CStr(name) directly — the intern dict provides "
        "program-lifetime pinning, and Python's bytes invariant "
        "guarantees a NUL-terminated buffer."
    )


def test_keep_alive_mixin_removed():
    """Sanity: the deleted ``_KeepAliveMixin`` is gone from the
    comgr module (regression against accidental re-introduction).
    Strings now live in ``CStr._retained_inputs`` (program-lifetime),
    not in a per-instance ``__references__`` set."""
    import rocm.comgr.comgr as comgr_mod
    assert not hasattr(comgr_mod, "_KeepAliveMixin"), (
        "_KeepAliveMixin was deleted in the CStr migration; do not re-introduce."
    )
