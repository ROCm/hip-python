# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Tests for the doxygen-first ``ptr_parm_intent`` rule in
``support.recipes.rocm.amdsmi``.

amdsmi.h tags 294 of 302 pointer parms with ``@param[in|out|in,out]``.
Trusting the doxygen tag resolves the 88 misclassifications the previous
verb-based heuristic produced (most prominent: every
``amdsmi_get_*(..., socket_count)`` style INOUT parm was mis-classified as
OUT).

These tests use synthetic single-function headers (no dependency on a real
amdsmi.h) so they run wherever pytest can. The ``test_real_amdsmi_*`` test
at the bottom is skipped without ``/opt/rocm/include/amd_smi/amdsmi.h``.
"""

import os

import pytest

from interfacegen import cython, treefactory
from interfacegen.cparser import CParser
from interfacegen.support.recipes import generic, rocm
from interfacegen.support.recipes.control import ParmIntent


def _build(header_text: str):
    parser = CParser("input.h", unsaved_files=[("input.h", header_text)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython, translation_unit=parser.translation_unit,
    )


def _parm(root, fname, pname):
    for n in root.walk(postorder=False):
        if isinstance(n, cython.Function) and n.name == fname:
            for p in n.parms:
                if p.name == pname:
                    return p
    raise KeyError(f"{fname}({pname})")


def test_doxygen_inout_overrides_get_verb_to_out():
    """`amdsmi_get_*` would default to OUT, but @param[in,out] wins."""
    root = _build("""
        /**
         *  @param[in] processor_handle a processor handle
         *  @param[in,out] utilization_counters caller allocates the array
         *  @param[in] count size of the array
         */
        int amdsmi_get_utilization_count(
            void *processor_handle,
            int *utilization_counters,
            unsigned int count
        );
    """)
    p = _parm(root, "amdsmi_get_utilization_count", "utilization_counters")
    assert rocm.amdsmi.ptr_parm_intent(p) == ParmIntent.INOUT


def test_doxygen_inout_overrides_set_verb_to_in():
    """`amdsmi_set_*` would default to IN, but @param[in,out] wins."""
    root = _build("""
        /**
         *  @param[in,out] utilization both an input baseline and updated.
         */
        int amdsmi_set_cpu_pwr_efficiency_mode(int *utilization);
    """)
    p = _parm(root, "amdsmi_set_cpu_pwr_efficiency_mode", "utilization")
    assert rocm.amdsmi.ptr_parm_intent(p) == ParmIntent.INOUT


def test_doxygen_out_overrides_handle_name_to_out():
    """Names ending in `_handle` would default to IN, but @param[out] wins.

    `void **node_handle` is a double-indirection slot, so the `[out]`
    refines to the callee-allocated flavor (the callee writes a fresh
    handle pointer); the coarse direction is still OUT.
    """
    root = _build("""
        /**
         *  @param[out] node_handle the new node handle is written here.
         */
        int amdsmi_get_node_handle(void **node_handle);
    """)
    p = _parm(root, "amdsmi_get_node_handle", "node_handle")
    verdict = rocm.amdsmi.ptr_parm_intent(p)
    assert verdict == ParmIntent.OUT_CALLEE_ALLOCATED
    assert verdict.direction == ParmIntent.OUT


def test_doxygen_out_overrides_inout_name_set():
    """`sensor_count` is in the INOUT-name set; @param[out] still wins."""
    root = _build("""
        /**
         *  @param[out] sensor_count number of sensors.
         */
        int amdsmi_get_supported_power_cap(unsigned int *sensor_count);
    """)
    p = _parm(root, "amdsmi_get_supported_power_cap", "sensor_count")
    assert rocm.amdsmi.ptr_parm_intent(p) == ParmIntent.OUT


def test_undocumented_parm_falls_back_to_verb_heuristic():
    """No @param tag → verb-based heuristic (`amdsmi_get_*` → OUT)."""
    root = _build("""
        /** Brief summary; intentionally no @param tags. */
        int amdsmi_get_undocumented_value(int *value);
    """)
    p = _parm(root, "amdsmi_get_undocumented_value", "value")
    assert rocm.amdsmi.ptr_parm_intent(p) == ParmIntent.OUT


def test_no_doc_comment_at_all_falls_back_to_verb_heuristic():
    """Function with no doxygen block → verb-based heuristic."""
    root = _build("int amdsmi_set_no_docs(int *value);")
    p = _parm(root, "amdsmi_set_no_docs", "value")
    assert rocm.amdsmi.ptr_parm_intent(p) == ParmIntent.IN


# --- Real-header sweep -----------------------------------------------------

AMDSMI_HEADER = "/opt/rocm/include/amd_smi/amdsmi.h"


@pytest.mark.skipif(
    not (os.path.exists(AMDSMI_HEADER)
         and os.path.exists("/opt/rocm/lib/llvm/lib/libclang.so")),
    reason="needs real amdsmi.h + libclang at /opt/rocm",
)
def test_real_amdsmi_doxygen_audit_zero_unintended_mismatches():
    """Audit every pointer parm in amdsmi.h: recipe verdict must match
    the doxygen `@param[…]` tag where one is present, EXCEPT for the
    handful of parms in `class amdsmi._MISTAGGED_OUT` where the upstream
    doxygen tag is wrong (overloaded `[in,out]` actually means `[out]`).

    With the doxygen-first rule in place, 0 unintended mismatches are
    expected. Without it, 88 mismatches are produced
    (utilization_counters, socket_count, processor_handles, etc.).
    The intentional override set is locked down here so any future
    addition is visible: when a new entry lands in `_MISTAGGED_OUT`,
    it must be reflected in the expected-mismatch set below.
    """
    import re

    parser = CParser(AMDSMI_HEADER)
    parser.parse()
    root = treefactory.from_libclang_translation_unit(
        backend=cython, translation_unit=parser.translation_unit,
    )

    # Reparse to build {fname: {pname: tag}}
    src = open(AMDSMI_HEADER).read()
    tag_re = re.compile(r"@param\[(in|out|in,\s*out)\]\s+([A-Za-z_]\w*)")
    fn_re = re.compile(
        r"\b([a-zA-Z_]\w*\s*\**)\s+(amdsmi_\w+)\s*\(([^;]*)\)\s*;",
        re.DOTALL,
    )
    docs = {}
    blocks = re.split(r"(/\*\*.*?\*/)", src, flags=re.DOTALL)
    for i, b in enumerate(blocks):
        if not b.startswith("/**"):
            continue
        params = tag_re.findall(b)
        if not params:
            continue
        after = blocks[i + 1] if i + 1 < len(blocks) else ""
        m = fn_re.search(after)
        if not m:
            continue
        fname = m.group(2)
        docs.setdefault(fname, {})
        for tag, pname in params:
            # `_DOXY_TAG_TO_INTENT` moved out of `class amdsmi` and into
            # the generic `documented_param_intent` rule when the
            # doxygen-intent helper was promoted to a chain rule. Read
            # from its new home.
            docs[fname][pname] = generic._DOXY_TAG_TO_INTENT[
                tag.replace(" ", "")
            ]

    # Build the intentional-override set keyed by (fname, pname) for
    # easy lookup. The set in `class amdsmi` is keyed by
    # (fname, parm_index) — re-key by name here so we don't have to
    # walk the tree twice.
    intentional = {}
    for n in root.walk(postorder=False):
        if not isinstance(n, cython.Function):
            continue
        for p in n.parms:
            if (n.name, p.parm_index) in rocm.amdsmi._MISTAGGED_OUT:
                intentional[(n.name, p.name)] = "OUT"

    mismatches = []
    for n in root.walk(postorder=False):
        if not isinstance(n, cython.Function):
            continue
        if not n.name.startswith("amdsmi_") or n.name not in docs:
            continue
        for p in n.parms:
            if not p.is_any_pointer:
                continue
            doc = docs[n.name].get(p.name)
            if doc is None:
                continue  # undocumented parm — verb fallback applies
            verdict = rocm.amdsmi.ptr_parm_intent(p)
            # Compare on the coarse direction: a scalar `@param[out]` may
            # refine to OUT_CALLEE_ALLOCATED, which still satisfies the
            # documented `[out]` direction (the allocation axis is
            # orthogonal and not expressed by the doxygen tag).
            if verdict.direction == doc:
                continue
            # An intentional override is allowed iff the recipe's
            # verdict matches the documented expected override.
            expected = intentional.get((n.name, p.name))
            if expected is not None and verdict.direction.name == expected:
                continue
            mismatches.append((n.name, p.name, doc.name, verdict.name))

    assert not mismatches, (
        f"{len(mismatches)} parms whose recipe verdict disagrees with the "
        f"doxygen tag (and not in the intentional override set).\nFirst 10:\n"
        + "\n".join(
            f"  {fn}({pn}): doc={doc} recipe={ver}"
            for fn, pn, doc, ver in mismatches[:10]
        )
    )
