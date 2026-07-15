# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Recipe test: the amdsmi recipe admits the foreign `struct timespec`.

`amdsmi_fabric_telemetry_dataset_t` has a BY-VALUE `timespec timestamp`
field (from `<time.h>`, transitively included by amdsmi.h). `timespec`
carries no `amdsmi_` prefix, so the strict-prefix `node_filter` would drop
it and the generated `cyamdsmi.pxd` would reference an undeclared bare
`timespec` (`'timespec' is not a type identifier`), while `amdsmi.pyx`
would call `timespec.fromPtr(...)` with no wrapper class.

Listing `"timespec"` in `amdsmi._EXTRA_TYPES` admits it, so the codegen
emits BOTH layers:
  - `cdef struct timespec:` in the cy module (a tagged struct, not a
    typedef, so it renders with `cdef struct` -> valid C `struct timespec`);
  - a `cdef class timespec` wrapper in the high-level module, matching the
    in-header by-value record precedent (`amdsmi_cper_timestamp_t`).
"""

import re

from interfacegen.support.recipes.rocm import amdsmi

from _codegen_helpers import build_root, find_record, make_generator, write_module


HEADER = """
struct timespec {
    long tv_sec;
    long tv_nsec;
};

typedef struct {
    struct timespec timestamp;
    unsigned int instance_count;
} amdsmi_fabric_telemetry_dataset_t;
"""


def test_amdsmi_node_filter_admits_timespec_record():
    """`amdsmi.node_filter` returns True for the foreign `struct timespec`
    (via `_EXTRA_TYPES`) while it has no `amdsmi_` prefix."""
    assert "timespec" in amdsmi._EXTRA_TYPES
    root = build_root(HEADER)
    ts = find_record(root, "timespec")
    assert amdsmi.node_filter(ts) is True, (
        "amdsmi.node_filter must admit `struct timespec` via _EXTRA_TYPES"
    )


def test_amdsmi_timespec_emitted_as_cdef_struct_with_wrapper(tmp_path):
    """End-to-end: with the real `amdsmi.node_filter`, the by-value
    `timespec` field resolves in the cy module as `cdef struct timespec:`
    (NOT `ctypedef struct`, which would emit an undeclared bare `timespec`
    in C) and gets a high-level wrapper class."""
    gen = make_generator(
        HEADER, module_name="mod_ts", node_filter=amdsmi.node_filter
    )
    files = write_module(gen, tmp_path)
    cy_pxd = files["cymod_ts.pxd"]
    pyx = files["mod_ts.pyx"]

    # C layer: a real `cdef struct timespec:` layout (valid C `struct
    # timespec`), and the dataset field references it.
    assert re.search(r"\bcdef\s+struct\s+timespec\s*:", cy_pxd), (
        f"expected `cdef struct timespec:` in cy pxd; full pxd:\n{cy_pxd}"
    )
    assert "ctypedef struct timespec" not in cy_pxd, (
        "timespec is a tagged struct, must not be emitted as `ctypedef "
        f"struct` (bare `timespec` is not valid C); full pxd:\n{cy_pxd}"
    )
    assert re.search(r"\btimespec\s+timestamp\b", cy_pxd), (
        f"expected the dataset to reference `timespec timestamp`; pxd:\n{cy_pxd}"
    )

    # High level: a wrapper class so `timespec.fromPtr(...)` resolves.
    assert re.search(r"\bcdef\s+class\s+timespec\b", pyx), (
        f"expected a `cdef class timespec` wrapper; full pyx:\n{pyx}"
    )
