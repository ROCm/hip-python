# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Shared scaffolding for the test_codegen_gap_*.py files.

Each gap test parses a tiny synthetic header, drives the cython backend, and
asserts on the emitted text or the type of failure. These helpers wrap the
boilerplate so each test stays focused on its specific pattern.
"""

import os

from interfacegen import cython, treefactory
from interfacegen.cparser import CParser
from interfacegen.support.recipes import control


def build_root(header_text: str):
    """Parse `header_text` as an in-memory header and return the cython tree
    root. Same shape as `_build_root` in `test_generic_recipes.py:75-79`.
    """
    parser = CParser("input.h", unsaved_files=[("input.h", header_text)])
    parser.parse()
    return treefactory.from_libclang_translation_unit(
        backend=cython, translation_unit=parser.translation_unit,
    )


def make_generator(
    header_text: str,
    *,
    module_name: str = "mod",
    node_filter=control.DEFAULT_NODE_FILTER,
    ptr_parm_intent=control.DEFAULT_PTR_PARM_INTENT,
    ptr_rank=control.DEFAULT_PTR_RANK,
    macro_type=lambda n: "int",
    renamer=None,
    util_pkg: str = "rocm.bindings.util",
    modifiers_lazy_loader: str = "",
    runtime_linking: bool = False,
    dll: str = None,
):
    """Build a CythonModuleGenerator backed by an in-memory header.

    Uses the ``header=(filename, content)`` tuple form supported by
    ``CythonModuleGenerator.__init__`` (see ``cython.py:3046-3066``), so the
    full ``write_module_files`` path is exercised without touching disk for
    the input header.

    ``modifiers_lazy_loader`` propagates to every ``Function`` node and
    drives whether the with-nogil emitter fires (any string containing
    ``"nogil"``) or the with-gil emitter (anything else, including the
    default empty string). Tests that assert on ``with nogil:`` shape
    must pass ``modifiers_lazy_loader=" noexcept nogil"``.
    """
    kw = dict(
        node_filter=node_filter,
        macro_type=macro_type,
        ptr_parm_intent=ptr_parm_intent,
        ptr_rank=ptr_rank,
        modifiers_lazy_loader=modifiers_lazy_loader,
    )
    if renamer is not None:
        kw["renamer"] = renamer
    return cython.CythonModuleGenerator(
        module_name,
        include_dir=None,
        header=("input.h", header_text),
        util_pkg=util_pkg,
        runtime_linking=runtime_linking,
        dll=dll,
        **kw,
    )


def write_module(generator, tmp_path) -> dict:
    """Run ``write_module_files`` into ``tmp_path`` and return
    ``{filename: text}`` for the four emitted files.
    """
    out_dir = str(tmp_path)
    os.makedirs(out_dir, exist_ok=True)
    generator.write_module_files(out_dir)
    out = {}
    for fname in os.listdir(out_dir):
        with open(os.path.join(out_dir, fname)) as fh:
            out[fname] = fh.read()
    return out


def find_function(root, name: str):
    for n in root.walk(postorder=False):
        if isinstance(n, cython.Function) and n.name == name:
            return n
    raise KeyError(name)


def find_record(root, name: str):
    for n in root.walk(postorder=False):
        if isinstance(n, (cython.Record, cython.Struct, cython.Union)):
            if n.name == name:
                return n
    raise KeyError(name)
