# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""`node_init` callbacks shared by the per-wheel generator modules.

A `node_init` runs for every node of a module's declaration tree and may
modify it, which is how the generators express wheel-level wiring that the
C-library recipes in `interfacegen.support.recipes.rocm` do not know about.
The status-enum override below is needed by every ROCm library that returns
a status enum and opts into the status-first-tuple contract, so it lives
here rather than in one of the `generators_*` modules.
"""

import interfacegen.tree


def make_status_node_init(prefix, status_type: str, success_const: str):
    """Per-Function override: drop ``except? <STATUS> nogil`` for functions
    whose return type is NOT ``<status_type>``, and prepend the status
    enum's success value as their first Python return value.

    ``prefix`` is either a single string or a tuple of strings (matched
    via ``startswith``); pass a tuple when the library uses several name
    prefixes (e.g. RCCL has ``ncclX`` AND ``pncclX`` profiling variants).

    The module-level ``modifiers_lazy_loader=" except? <STATUS_INTERNAL_ERROR>
    nogil"`` only type-checks for functions that actually return the status
    enum. Three families of functions need the override:

    1. **Non-enum returns** — ``hipblasStatusToString`` returns ``const char
       *``, ``hipfftMakePlan`` helpers may return ``int`` / ``size_t``, etc.
       Cython rejects ``except?`` because the sentinel value type doesn't
       match the return type.
    2. **Different-enum returns** — ``hipsparseGetMatType`` returns
       ``hipsparseMatrixType_t`` (NOT ``hipsparseStatus_t``). Same
       type-mismatch problem; ``is_enum`` alone wouldn't catch this.
    3. **void returns** — ``noexcept`` is the only valid modifier.

    For all of these, ``noexcept nogil`` is the right modifier — the
    ``nogil`` declaration still applies (so ``with nogil:`` blocks at call
    sites are valid) but no exception-translation watcher is inserted.

    In addition, because the module opts into
    ``python_interface_always_return_tuple`` (status-first-tuple contract),
    these non-status functions get ``<status_type>.<success_const>``
    prepended as their first return value via
    ``prepend_python_return_value`` so callers can always unpack
    ``status, *rest = fn(...)``. ``<status_type>`` resolves to the Python
    IntEnum class (either named directly or aliased from its tag name).

    Same overall pattern as ``hip_node_init`` / ``hiprtc_node_init`` in
    ``generators_hip.py``.
    """
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)

    def _init(node):
        if isinstance(node, interfacegen.tree.Function):
            if not node.name.startswith(prefixes):
                return
            # Check the function's return-type cython spelling; if it is
            # not the status enum we need ``noexcept`` instead of
            # ``except? <STATUS>``.
            try:
                return_typename = node.cython_global_typename
            except Exception:
                return_typename = ""
            if return_typename != status_type:
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
                node.prepend_python_return_value(
                    f"{status_type}.{success_const}",
                    status_type,
                    f"Always returns `~.{status_type}.{success_const}`.",
                )

    return _init


def chain_node_init(*inits):
    """Run several ``node_init`` callbacks over every node, in order.

    A module gets one ``node_init``, so a generator that already overrides
    function modifiers cannot also flag fields without composing the two.
    """

    def _init(node):
        for init in inits:
            init(node)

    return _init
