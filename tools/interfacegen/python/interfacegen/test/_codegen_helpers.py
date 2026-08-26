# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

"""Shared scaffolding for the test_codegen_gap_*.py files.

Each gap test parses a tiny synthetic header, drives the cython backend, and
asserts on the emitted text or the type of failure. These helpers wrap the
boilerplate so each test stays focused on its specific pattern.
"""

import os
import subprocess
import sys

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
        backend=cython,
        translation_unit=parser.translation_unit,
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
    typedef_aliases: dict = None,
    typedef_specs: dict = None,
    node_init=None,
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

    ``node_init`` is the per-node hook recipes use to override those
    modifiers function by function; it runs after the module-wide
    defaults have been applied.
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
    if node_init is not None:
        kw["node_init"] = node_init
    if typedef_aliases is not None:
        kw["typedef_aliases"] = typedef_aliases
    if typedef_specs is not None:
        kw["typedef_specs"] = typedef_specs
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
        # The emitters write UTF-8 (doxygen comments carry non-ASCII
        # punctuation); reading back with the locale codec fails on Windows.
        with open(os.path.join(out_dir, fname), encoding="utf-8") as fh:
            out[fname] = fh.read()
    return out


def cython_check(tmp_path, module_name: str = "mod"):
    """Compile a module that cimports the emitted declarations, and return the
    ``CompletedProcess``. Call after ``write_module``.

    Naming ``cy<module>.pxd`` on the command line instead does not work, and
    the output gives no hint why: Cython compiles the named file as a module
    body *and* loads that same file as the module's declarations, so every
    symbol arrives twice and a few hundred spurious "Non-extern C function
    declared but not defined" errors bury whatever was real. A cimport reaches
    the .pxd in the role it was emitted for.
    """
    out_dir = str(tmp_path)
    with open(os.path.join(out_dir, "_probe.pyx"), "w") as fh:
        fh.write(f"cimport cy{module_name}\n")
    return subprocess.run(
        [sys.executable, "-m", "cython", "--3str", "_probe.pyx"],
        cwd=out_dir,
        capture_output=True,
        text=True,
    )


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
