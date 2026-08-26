# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Gap test: a type an admitted node depends on is dropped for lacking the prefix.

Real-world hit (from amdsmi.h:2036, 2231 and 4805):

    typedef struct { ... } amd_metrics_table_header_t;   // no `amdsmi_` prefix

    typedef struct {
      amd_metrics_table_header_t common_header;          // by-value field
      ...
    } amdsmi_gpu_metrics_t;

    amdsmi_status_t amdsmi_get_gpu_metrics_header_info(
        amdsmi_processor_handle processor_handle,
        amd_metrics_table_header_t *header_value);

The amdsmi recipe's `node_filter` admits `amdsmi_*` / `AMDSMI_*` and nothing
else, so the header type is dropped while the struct field and the parameter
that name it are kept. The emitted module then references a type it never
defines, and Cython rejects it:

    cyamdsmi.pxd:1204:8:  'amd_metrics_table_header_t' is not a type identifier
    cyamdsmi.pxd:3161:80: 'amd_metrics_table_header_t' is not a type identifier

The recipe compensates by listing the type in
``support.recipes.rocm.amdsmi._EXTRA_TYPES``, alongside ``processor_type_t``
and ``timespec``. That set is the workaround, not the fix: admission does not
follow references, so every recipe has to enumerate its non-prefixed
dependencies by hand and keep doing so as headers change. Deleting
``_EXTRA_TYPES`` is what closing this gap would allow.
"""

import re

import pytest
from _codegen_helpers import cython_check, make_generator, write_module
from interfacegen.tree import MacroDefinition

HEADER = """
typedef struct {
    unsigned int structure_size;
    unsigned char format_revision;
} amd_metrics_table_header_t;

typedef struct {
    amd_metrics_table_header_t common_header;
    unsigned long accumulation_counter;
} amdsmi_gpu_metrics_t;

int amdsmi_get_gpu_metrics_header_info(amd_metrics_table_header_t *header_value);
"""

# What the recipe hand-carries because the prefix cannot express it.
EXTRA_TYPES = frozenset({"amd_metrics_table_header_t"})


def _prefix_only_filter(node):
    """Mimics amdsmi's `node_filter` with `_EXTRA_TYPES` dropped."""
    if isinstance(node, MacroDefinition):
        return False
    return (node.name or "").startswith("amdsmi_")


def _prefix_plus_extras_filter(node):
    """Mimics amdsmi's `node_filter` as it ships, `_EXTRA_TYPES` included."""
    if isinstance(node, MacroDefinition):
        return False
    name = node.name or ""
    return name.startswith("amdsmi_") or name in EXTRA_TYPES


def _emit(tmp_path, node_filter, module_name):
    gen = make_generator(
        HEADER,
        module_name=module_name,
        node_filter=node_filter,
        runtime_linking=True,
        dll="libfake.so",
    )
    return write_module(gen, tmp_path)[f"cy{module_name}.pxd"]


def test_nonprefixed_dependency_emitted_when_admitted(tmp_path):
    """Admitting the dependency explicitly produces a module that compiles.

    This is the control for the strict-xfail below, which needs one: a failing
    assertion inside an xfail test is indistinguishable from the expected
    failure, so nothing there can report that the compile step itself broke.
    """
    pxd = _emit(tmp_path, _prefix_plus_extras_filter, "mod_extras")
    assert re.search(
        r"\bctypedef\s+struct\s+amd_metrics_table_header_t\b", pxd
    ), f"expected the dependency's own definition; full pxd:\n{pxd}"

    res = cython_check(tmp_path, "mod_extras")
    assert res.returncode == 0, (
        f"the admitted case must compile:\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )


def test_nonprefixed_dependency_dropped_when_unadmitted(tmp_path):
    """Prefix-only admission emits the references but not the definition.

    Characterises today's behaviour rather than endorsing it. When admission
    learns to follow references this inverts, and the xfail below flips.
    """
    pxd = _emit(tmp_path, _prefix_only_filter, "mod_prefix")

    # Both reference sites survive: the by-value field and the parameter.
    assert re.search(
        r"amd_metrics_table_header_t\s+common_header", pxd
    ), f"expected the field to reference the dependency; full pxd:\n{pxd}"
    assert re.search(
        r"amd_metrics_table_header_t\s*\*", pxd
    ), f"expected the parameter to reference the dependency; full pxd:\n{pxd}"
    # The definition does not.
    assert not re.search(
        r"\bctypedef\s+struct\s+amd_metrics_table_header_t\b", pxd
    ), f"the dependency is defined after all -- gap closed?; full pxd:\n{pxd}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Admission does not follow references, so a type an admitted node "
        "depends on is dropped for lacking the prefix. Flip, and delete the "
        "amdsmi recipe's _EXTRA_TYPES, once it does."
    ),
)
def test_prefix_only_filtering_compiles(tmp_path):
    _emit(tmp_path, _prefix_only_filter, "mod_gap")
    res = cython_check(tmp_path, "mod_gap")
    assert res.returncode == 0, (
        f"prefix-only admission does not compile:\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
