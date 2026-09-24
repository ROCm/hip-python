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


"""Cython backend — backend submodule.

Carved out of the historical interfacegen.cython monolith.
Public callers should import from `interfacegen.cython`,
not from this submodule directly.
"""

__author__ = "Advanced Micro Devices, Inc."

import ctypes
import keyword
import logging
import os
import re
import sys
import textwrap
import typing

import clang.cindex

from .. import cparser, cythontemplates, doxyparser, tree, typerender
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults, _doxygen, _entities, _function, _mixins
from ._defaults import *  # noqa: F401,F403
from ._doxygen import *  # noqa: F401,F403
from ._entities import *  # noqa: F401,F403
from ._function import *  # noqa: F401,F403
from ._mixins import *  # noqa: F401,F403

__all__ = [
    "CythonBackend",
    "CythonModuleGenerator",
]


def _check_typedef_maps(typedef_aliases: dict, typedef_specs: dict):
    """Reject per-module width declarations that cannot be resolved.

    An entry that silently resolves to nothing is worse than no entry at all:
    the typedef keeps the codegen host's canonical spelling and the wrapper
    dispatch keeps picking a wrapper by the host's data model, which is exactly
    the mismatch these maps exist to remove. So a recipe typo fails loudly here
    rather than in a shipped ``.pxd``.
    """
    for name, target in (typedef_aliases or {}).items():
        if target not in typerender.FIXED_WIDTH_INT_SPECS:
            raise ValueError(
                f"typedef_aliases['{name}']: '{target}' is not a known "
                "fixed-width integer typedef "
                f"({sorted(typerender.FIXED_WIDTH_INT_SPECS)})"
            )
    for name, spec in (typedef_specs or {}).items():
        if tuple(spec) not in typerender.FIXED_WIDTH_INT_SPELLINGS:
            raise ValueError(
                f"typedef_specs['{name}']: {spec} is not a known "
                "(signed, bits) integer spec "
                f"({sorted(typerender.FIXED_WIDTH_INT_SPELLINGS)})"
            )
    both = set(typedef_aliases or ()) & set(typedef_specs or ())
    if both:
        raise ValueError(
            "typedefs declared in both 'typedef_aliases' and "
            f"'typedef_specs': {sorted(both)}"
        )


class CythonBackend:
    def from_libclang_translation_unit(
        translation_unit: clang.cindex.TranslationUnit,
        filename: str,
        util_pkg: str,
        warn_mode: control.Warnings = control.Warnings.IGNORE,
        module_opts: dict = None,
        **opts,
    ):
        """See `CythonBackend.__init__` for further details."""
        from interfacegen import treefactory

        root = treefactory.from_libclang_translation_unit(
            backend=sys.modules[__name__],  # this module is the backend
            translation_unit=translation_unit,
            warn_mode=warn_mode,
        )
        return CythonBackend(
            root,
            filename,
            util_pkg,
            module_opts=module_opts,
            **opts,
        )

    def __init__(
        self,
        root,
        filename: str,
        util_pkg: str = "",
        modifiers_lazy_loader: str = "",
        error_return_value_lazy_loader: str = None,
        node_filter: callable = control.DEFAULT_NODE_FILTER,
        macro_type: callable = DEFAULT_MACRO_TYPE,
        ptr_parm_intent: callable = control.DEFAULT_PTR_PARM_INTENT,
        ptr_rank: callable = control.DEFAULT_PTR_RANK,
        ptr_complicated_type_handler=None,
        renamer: callable = DEFAULT_RENAMER,
        raw_comment_cleaner: callable = DEFAULT_RAW_COMMENT_CLEANER,
        docstring_cleaner: callable = DEFAULT_DOCSTRING_CLEANER,
        node_init: callable = lambda node: None,
        module_opts: dict = None,
        typedef_aliases: dict = None,
        typedef_specs: dict = None,
    ):
        """Constructor.

        Args:
            node_filter (callable, optional):
                Filter for selecting the nodes to include in generated output. Defaults to ``lambda x: True``.
                Note that other callbacks are applied to non-filtered nodes also.
            node_init (callable, optional):
                Callback that can be used to modify certain nodes arbitrarily.
                Note that this callback is applied after all other callbacks (excluding ``node_filter``).
                Defaults to no-operation.
            modifiers_lazy_loader (str, optional):
                Modifiers for the lazy loader function. Defaults to " except *", i.e.
                there is always a check for exceptions performed.
                This argument must be specified together with `error_return_value_lazy_loader`.
                More details on Cython exception handling:
                https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
            error_return_value_lazy_loader (str, optional):
                Designated error return value for the lazy loader interface.
                Empty string and 'None' indicate no return value is used to return errors. Defaults to None.
                This argument must be specified together with `modifiers_lazy_loader`.
                More details on Cython exception handling:
                https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
            macro_type (callable, optional):
                Assigns a type to a macro node. Defaults to ``lambda x: "int"``.
            ptr_parm_intent (callable, optional):
                Assigns the intent (in,out,inout,create) to a pointer-type function parameter/struct field node..
            ptr_rank (callable, optional):
                Assigns the "rank" (scalar,buffer) to a function parameter node.
            ptr_complicated_type_handler (callable, optional):
                A handler that infers a type for complicated pointer types.
                Selects `CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(f"{util_pkg}.types.")`
                if the default value `None` is not overwritten with a user callback.
            typedef_aliases (dict, optional):
                Maps library typedefs that pin a width but are not part of
                ``<stdint.h>`` to the stdint name denoting the same C type
                everywhere, e.g. ``{"hoff_t": "int64_t"}``. Without an entry
                such a typedef is canonicalized to the codegen host's spelling
                (``long`` on LP64, ``long long`` on LLP64), which no
                hand-written consumer can then name portably. See
                ``interfacegen.typerender.fixed_width_typedef``.
            typedef_specs (dict, optional):
                The same fact stated directly, as ``(signed, bits)``, e.g.
                ``{"hoff_t": (True, 64)}``. The stdint typedef denoting that
                width is emitted, so an entry here pins the spelling as well as
                the width. Use ``typedef_aliases`` when an existing stdint name
                says it more readably; a typedef must not appear in both maps.
        Note:
            Argument 'root' has no type hint in order to prevent a circular inclusion error.
            Instead an assertion is used in the body that checks if the type is `tree.Root`.
        """

        self.root = root
        self.filename = filename
        self.util_pkg = util_pkg
        self.module_opts = dict(module_opts) if module_opts else {}
        self.module_opts.setdefault(
            "python_interface_always_return_tuple", False
        )
        self.modifiers_lazy_loader = modifiers_lazy_loader
        self.error_return_value_lazy_loader = error_return_value_lazy_loader
        self.node_filter = node_filter
        self.macro_type = macro_type
        self.ptr_parm_intent = ptr_parm_intent
        self.ptr_rank = ptr_rank
        if ptr_complicated_type_handler is None:
            self.ptr_complicated_type_handler = (
                CREATE_DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(
                    f"{util_pkg}.types."
                )
            )
        else:
            self.ptr_complicated_type_handler = ptr_complicated_type_handler
        self.renamer = renamer
        self.raw_comment_cleaner = raw_comment_cleaner
        self.docstring_cleaner = docstring_cleaner
        self.node_init = node_init
        _check_typedef_maps(typedef_aliases, typedef_specs)
        self.typedef_aliases = (
            dict(typedef_aliases) if typedef_aliases else None
        )
        self.typedef_specs = (
            {n: tuple(s) for n, s in typedef_specs.items()}
            if typedef_specs
            else None
        )

        self.initialize_nodes()

    def initialize_nodes(self):
        """Initializes all nodes.

        Note:
            Post-order walk, touches nested struct/union/enum declarations before their
            parent.
        """
        for node in self.root.walk(postorder=True):
            if isinstance(node, CythonMixin):
                # set defaults
                setattr(node, "util_types_prefix", self.util_pkg + ".types.")
                setattr(node, "sep", "_")
                # set user callbacks
                setattr(node, "renamer", self.renamer)
                setattr(node, "typedef_aliases", self.typedef_aliases)
                setattr(node, "typedef_specs", self.typedef_specs)
                setattr(node, "raw_comment_cleaner", self.raw_comment_cleaner)
                setattr(node, "docstring_cleaner", self.docstring_cleaner)
                if isinstance(node, MacroDefinition):
                    setattr(node, "macro_type", self.macro_type)
                elif isinstance(node, Field):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                elif isinstance(node, Parm):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(node, "ptr_intent", self.ptr_parm_intent)
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                    # Make the recipe filter reachable from the
                    # parm at render time. Used by the
                    # pointer-to-record code path in `_function.py`
                    # to decide whether the innermost record is
                    # actually emitted as a per-type wrapper class
                    # — if not (foreign-prefix record like libc
                    # `FILE` / `_IO_FILE`), the renderer falls
                    # back to the handler-driven generic wrapper
                    # (default: `rocm.bindings.util.types.Pointer`).
                    setattr(node, "node_filter", self.node_filter)
                elif isinstance(node, Typedef):
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                elif isinstance(node, Function):
                    setattr(node, "ptr_rank", self.ptr_rank)
                    setattr(
                        node,
                        "ptr_complicated_type_handler",
                        self.ptr_complicated_type_handler,
                    )
                    node.modifiers_lazy_loader = self.modifiers_lazy_loader
                    node.error_return_value_lazy_loader = (
                        self.error_return_value_lazy_loader
                    )
                self.node_init(node)

    def _transitively_admitted_records(self):
        """Records that the user `node_filter` rejects but that are
        referenced by an admitted ``Typedef``'s underlying type and
        live in the same source file as that typedef.

        Without this, ``typedef struct _Tag { ... } T;`` where the user
        filter rejects ``_Tag`` (e.g. because it starts with ``_``) but
        accepts ``T`` produces a dangling ``ctypedef _Tag T`` with no
        backing ``cdef struct _Tag:`` block — Cython then errors with
        ``'_Tag' is not a type identifier``. We close the gap by
        admitting any ``Record`` whose name appears as the
        ``Typedef.typeref`` of an admitted typedef AND whose source
        location is in the same header.

        The same-file gate is critical: cross-file typedef chains like
        ``typedef hipComplex hipfftComplex`` (where the underlying
        ``float2`` Record lives in ``amd_hip_complex.h``, not
        ``hipfft.h``) must NOT be admitted — the downstream module
        already cimports those types from a sibling binding package
        (``from rocm.bindings.hip cimport float2``), so admitting them
        here would cause a duplicate ``cdef class float2`` definition.

        Returns the set of ``Record.name`` values to admit transitively.
        """
        from .. import tree

        wanted = set()
        for node in self.root.walk(postorder=True):
            if isinstance(node, tree.Typedef) and self.node_filter(node):
                ref = getattr(node, "typeref", None)
                if isinstance(ref, tree.Record) and not self.node_filter(ref):
                    if ref.name and self._same_source_file(node, ref):
                        wanted.add(ref.name)
        return wanted

    @staticmethod
    def _same_source_file(a, b) -> bool:
        """True if both nodes' clang cursors come from the same header
        (or both from no file at all). Used by
        ``_transitively_admitted_records`` to avoid pulling in Records
        from sibling-package headers that the consumer already cimports."""
        fa = getattr(a, "file", None)
        fb = getattr(b, "file", None)
        return fa == fb and fa is not None

    @staticmethod
    def _topmost_ancestor(node):
        """Walk ``node.parent`` up to the outermost non-Root ancestor.

        For a top-level node this returns the node itself; for a nested
        type (a struct/union/enum nested in a record, or an inline
        function-pointer field/param) it returns the enclosing top-level
        Record/Function/Typedef. Used to let nested declarations inherit
        their top-most enclosing declaration's admission verdict — see
        ``walk_filtered_nodes``.
        """
        from .. import tree

        top = node
        while top.parent is not None and not isinstance(top.parent, tree.Root):
            top = top.parent
        return top

    def walk_filtered_nodes(self):
        """Walks the filtered nodes in post-order and sets the renamer of each node.

        Note:
            Post-order walk, touches nested struct/union/enum declarations before their
            parent.

            Records named as the underlying type of an admitted Typedef
            are admitted transitively even if the user `node_filter`
            rejects them — see ``_transitively_admitted_records`` for
            the rationale (leading-underscore tag names like
            ``_hipblasLtMatmulAlgo_t``).

            Nested declarations — a struct/union/enum nested inside a
            record, or an inline ``AnonymousFunctionPointer`` field/param
            — are admitted transitively whenever their TOP-MOST enclosing
            declaration is admitted. Their synthesized names
            (``<parent>_struct_<N>``, ``<parent>_union_<N>``,
            ``<parent>_anon_funptr_<N>``) carry no library prefix and so
            never match a strict-prefix recipe filter, but the enclosing
            decl's rendering references those names as field/param types —
            the matching ``cdef struct``/``ctypedef`` MUST therefore be
            emitted, otherwise Cython sees an undeclared identifier ("not
            a type identifier", or a fallback to "Python object" inference
            that then fails under ``nogil``). Walking to the top-most
            ancestor (rather than the immediate parent) makes this work at
            arbitrary nesting depth, e.g. a union field whose type nests a
            further anonymous struct, or a function pointer inside a
            nested anonymous record. hipFILE's ``hipFileDescr.handle`` and
            HSA's ``hsa_iterate_agents`` exhibit these shapes.
        """
        from .. import tree

        transitive_records = self._transitively_admitted_records()
        for node in self.root.walk(postorder=True):
            if isinstance(node, CythonMixin):
                if not isinstance(node, (Field, Parm, Root)):
                    admitted = self.node_filter(node)
                    if (
                        not admitted
                        and isinstance(node, tree.Record)
                        and node.name in transitive_records
                    ):
                        admitted = True
                    # Nested records/enums and inline anonymous function
                    # pointers inherit the admission of their top-most
                    # enclosing declaration. ``top is not node`` excludes
                    # top-level nodes (a rejected top-level type stays
                    # rejected); the walk handles arbitrary nesting depth.
                    if not admitted and isinstance(
                        node,
                        (
                            tree.Record,
                            tree.Enum,
                            tree.AnonymousFunctionPointer,
                        ),
                    ):
                        top = self._topmost_ancestor(node)
                        if top is not node and self.node_filter(top):
                            admitted = True
                    if admitted:
                        _log.debug(
                            f" touch {node.__class__.__name__} {node.name} from {node.cursor.kind} {node.cursor.spelling} ({node.render_location()})"
                        )
                        yield node

    def walk_entities_to_import(self, cmodule: bool):
        """Yields the entities that need to imported in the c-prefixed Cython module
        or the Python module.

        For the Python module, yields all top-level nodes aside from FunctionPointer and Record nodes (structs and unions) as those are modelled
        as Cython extension classes ("cdef class"). Yields nothing for the c-prefixed Cython module.

        Note:
            Utilizes `walk_filtered_nodes(self)`, i.e. the result depends on the
            supplied node filter.

        Args:
            cmodule (bool):
                If we perform this operation for the c-prefixed Cython module.
                In this case nothing is yielded at all.
        """

        if cmodule:
            yield from ()
        else:
            for node in self.walk_filtered_nodes():
                if isinstance(node, (tree.FunctionPointer, tree.Record)):
                    continue
                if isinstance(node, Typedef):
                    if not node.emits_python_alias():
                        continue
                yield node

    def walk_entities_to_cimport(self, cmodule: bool) -> CythonMixin:
        """Yields the entities that need to c-imported in the c-prefixed Cython module
        or the Python module.

        For the Python module, yields only FunctionPointer and Record nodes (structs and unions) as those are modelled
        as Cython extension classes ("cdef class").
        Yields all top-level nodes for the c-prefixed Cython module.

        Note:
            Utilizes `walk_filtered_nodes(self)`, i.e. the result depends on the
            supplied node filter.

        Args:
            cmodule (bool):
                If we perform this operation for the c-prefixed Cython module.
                In this case all top-level nodes are yielded.
        """

        if cmodule:
            yield from self.walk_filtered_nodes()
        else:
            for node in self.walk_filtered_nodes():
                if isinstance(node, (tree.FunctionPointer, tree.Record)):
                    yield node

    def create_c_interface_decl_part(self, runtime_linking: bool = False):
        """Returns the content of a Cython bindings file.

        Creates the content of a Cython bindings file.

        Contains Cython declarations per C declaration plus declarations of
        hardcoded macro values and helper types that have been introduced for
        nested enum/struct/union types

        Note:
            Anonymous types for which we have a tree node with
            autogenerated name must be excluded from the `extern from "<header_name.h>`
            block as entities listed within the body of the construct,
            are assumed by Cython to be present in C code whenever
            the respective header is included.
            The same holds true for records (struct, union) and enum types that
            are nested inside of another record.

            Moving those entities out of the `extern from` block
            ensures that Cython creates a proper C type on its own.
        """

        global indent
        curr_indent = ""
        result = []

        last_was_extern = False
        for node in self.walk_filtered_nodes():
            if (
                (runtime_linking and isinstance(node, Function))
                or isinstance(node, AnonymousFunctionPointer)
                or (
                    isinstance(
                        node,
                        (
                            tree.AnonymousEnum,
                            tree.AnonymousStruct,
                            tree.AnonymousUnion,
                        ),
                    )
                    and node.is_cursor_anonymous
                )
                or isinstance(node, ParentIsRecordMixin)
                and node.parent_is_record
                or (
                    isinstance(node, MacroDefinition)
                    and (
                        node.no_right_hand_side
                        or node.interpret_right_hand_side_as_str
                        or node.hardcoded_right_hand_side
                    )
                )
            ):
                if isinstance(node, Function):
                    contrib = node.render_cython_lazy_loader_decl()
                else:
                    contrib = node.render_c_interface_decl()
                curr_indent = ""
                last_was_extern = False
            else:
                # NOTE: these declarations must be in `cdef extern from ...` environment
                if not last_was_extern:
                    result.append(f'cdef extern from "{self.filename}":')
                else:
                    pass  # NOTE: already in 'cdef extern from ...' environment
                curr_indent = indent
                contrib = node.render_c_interface_decl()
                last_was_extern = True
            if contrib:
                result.append(textwrap.indent(contrib, curr_indent))
        return result

    def create_c_interface_impl_part(
        self, dll: str, util_pkg: str, module_name: str
    ):
        result = []
        lib_handle = "_lib_handle"
        # Module-unique name for the only pxd-exported helper. Prevents the
        # `cimport *` leak/collision that made derived modules' `has_symbol`
        # bind to a parent module's `__has_symbol` (see design note in
        # `render_c_interface_decl_part`).
        has_symbol = f"__{module_name}_has_symbol"
        # The DLL stem used to resolve the path lazily via
        # rocm.bindings.util.paths.get_library_path. Strip a trailing ".so"/.dll
        # so e.g. "libamdhip64.so" becomes "amdhip64".
        dll_stem = dll
        for _suffix in (".so", ".dll", ".dylib"):
            if dll_stem.endswith(_suffix):
                dll_stem = dll_stem[: -len(_suffix)]
                break
        if dll_stem.startswith("lib"):
            dll_stem = dll_stem[len("lib") :]
        result.append(
            textwrap.dedent(
                f"""\
            cimport {util_pkg}.loader as loader
            cdef void* {lib_handle} = NULL

            cdef bytes _dll_path = b""  # Cached path, computed on first access

            cdef int __init() except 1 nogil:
                global _dll_path, {lib_handle}
                cdef char* dll = NULL
                if {lib_handle} == NULL:
                    with gil:
                        # Lazy path resolution - only happens on first function call
                        if not _dll_path:
                            from rocm.bindings.util.paths import get_library_path
                            _dll_path = get_library_path('{dll_stem}')
                        dll = _dll_path
                    return loader.open_library(&{lib_handle}, dll)
                return 0

            cdef int __init_symbol(void** result, const char* name) except 1 nogil:
                global {lib_handle}
                cdef int init_result = 0
                if {lib_handle} == NULL:
                    init_result = __init()
                    if init_result > 0:
                        return init_result
                if result[0] == NULL:
                    return loader.load_symbol(result,{lib_handle}, name)
                return 0

            cdef bint {has_symbol}(const char* name) noexcept nogil:
                # Non-raising symbol-presence probe. Lazy-loads the DLL
                # the same way __init_symbol does, then asks the loader
                # whether the symbol exists. Returns False on any DLL
                # open failure or missing symbol — never raises. Used by
                # the python-visible `has_symbol(name)` wrapper in the
                # high-level module.
                global {lib_handle}
                cdef int init_result = 0
                if {lib_handle} == NULL:
                    with gil:
                        try:
                            init_result = __init()
                        except Exception:
                            return False
                    if init_result > 0:
                        return False
                return loader.has_symbol({lib_handle}, name)
            """
            )
        )
        for node in self.walk_filtered_nodes():
            if isinstance(node, Function):
                result.append("\n" + node.render_cython_lazy_loader_def())
            elif isinstance(node, MacroDefinition):
                contrib = node.render_c_interface_impl()
                if contrib:
                    result.append("\n" + contrib)
        return result

    def render_c_interface_decl_part(
        self, runtime_linking: bool = False, module_name: str = None
    ):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n\n"
        parts = list(self.create_c_interface_decl_part(runtime_linking))
        # Public declaration for the cdef helper that the high-level
        # python module's `has_symbol(name)` wrapper cimports. Only
        # emitted under runtime_linking — the matching impl in
        # `create_c_interface_impl_part` is gated the same way (the
        # `__init`/`__init_symbol`/`__<module>_has_symbol` helpers all
        # live in the runtime-linking prologue). The helper is given a
        # module-unique name so it can never leak/collide through
        # `cimport *`.
        if runtime_linking:
            if module_name is None:
                raise ValueError(
                    "argument 'module_name' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            parts.append(
                f"cdef bint __{module_name}_has_symbol(const char* name) noexcept nogil"
            )
        return nl.join(parts)

    def render_c_interface_impl_part(
        self,
        util_pkg: str,
        runtime_linking: bool = False,
        dll: str = None,
        module_name: str = None,
    ):
        """Returns the Cython bindings file content for the given headers."""
        nl = "\n"
        if runtime_linking:
            if dll is None:
                raise ValueError(
                    "argument 'dll' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            if module_name is None:
                raise ValueError(
                    "argument 'module_name' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            return nl.join(
                self.create_c_interface_impl_part(dll, util_pkg, module_name)
            )
        else:
            return ""

    def create_python_interface_decl_part(self, cmodule):
        """Renders Python interfaces in Cython."""

        result = []
        cprefix = f"{cmodule}."
        for node in self.walk_filtered_nodes():
            contrib = node.render_python_interface_decl(cprefix=cprefix)
            if contrib is not None:
                result.append(contrib)
        return result

    def create_python_interface_impl_part(self, cmodule):
        """Renders Python interfaces in Cython."""

        result = []
        cprefix = f"{cmodule}."
        # Per-render-pass opts: configuration from the CythonModuleGenerator
        # plus freshly allocated accumulator lists. Passed as an explicit
        # kwarg into each node's render_python_interface_impl — no setattr
        # injection, no class-attribute side effects.
        module_opts = dict(self.module_opts)
        module_opts["all"] = []  # required to define order
        module_opts["docstring_attributes"] = []
        for node in self.walk_filtered_nodes():
            contrib = node.render_python_interface_impl(
                cprefix=cprefix,
                module_opts=module_opts,
            )
            if contrib is not None:
                result.append(contrib)
        return (
            result,
            module_opts["docstring_attributes"],
            module_opts["all"],
        )

    def render_python_interface_decl_part(self, cython_c_bindings_module: str):
        """Returns the Python interface file content for the given headers."""
        result = self.create_python_interface_decl_part(
            cython_c_bindings_module
        )
        nl = "\n\n"
        return f"""\
{nl.join(result)}"""

    def render_python_interface_impl_part(
        self,
        cython_c_bindings_module: str,
        module_name: str = None,
        runtime_linking: bool = False,
    ):
        """Returns the Python interface file content for the given headers."""
        contribs, docstring_attributes, all = (
            self.create_python_interface_impl_part(cython_c_bindings_module)
        )
        prefix_parts = []
        if runtime_linking:
            if module_name is None:
                raise ValueError(
                    "argument 'module_name' must not be 'None' if 'runtime_linking' is set to 'True'"
                )
            # Python-visible `has_symbol(name)` wrapper. Probes whether
            # the runtime-linked DLL exports the named symbol — useful
            # for feature detection against libraries that ship in two
            # flavours (e.g. a stripped system libLLVM.so vs a
            # static-archive aggregate). Delegates to the cy*-level
            # `__<module>_has_symbol` cdef helper (declared in the matching
            # cy*.pxd; see `render_c_interface_decl_part`) which
            # handles lazy DLL initialisation and never raises.
            prefix_parts.append(
                textwrap.dedent(
                    f"""\
                def has_symbol(name) -> bool:
                    r\"\"\"Probe whether the runtime-linked DLL exports a symbol.

                    Non-raising. Lazily opens the DLL on first call (same
                    resolution path as the per-symbol lazy loaders), then
                    asks the loader whether the named symbol is present.
                    Returns ``False`` if the DLL fails to open or the symbol
                    isn't there.

                    Args:
                        name: Symbol name to probe (``str`` or ``bytes``).

                    Returns:
                        True if the symbol resolves, False otherwise.
                    \"\"\"
                    cdef bytes name_bytes
                    if isinstance(name, str):
                        name_bytes = name.encode("utf-8")
                    elif isinstance(name, (bytes, bytearray)):
                        name_bytes = bytes(name)
                    else:
                        raise TypeError("name must be str, bytes, or bytearray")
                    return {cython_c_bindings_module}.__{module_name}_has_symbol(<const char*>name_bytes)
                """
                )
            )
            all = list(all) + ["has_symbol"]
        result = (
            ("\n\n".join(prefix_parts) + "\n\n" if prefix_parts else "")
            + "\n\n".join(contribs).rstrip()
            + "\n\n"
            + "__all__ = [\n"
            + "\n".join([f'    "{e}",' for e in all])
            + "\n]"
        )
        return (result, docstring_attributes)


class CythonModuleGenerator:
    """Generate Cython extension modules for a HIP C interface.

    Generates Cython extension modules for a HIP C interface
    based on a list of header file names and the name of
    a library to link.
    """

    def __init__(
        self,
        global_module_name: str,
        include_dir: str,
        header: str,
        util_pkg: str,
        runtime_linking: bool = False,
        dll: str = None,
        cflags=[],
        module_opts: dict = None,
        **opts,
    ):
        r"""Constructor.

        Args:
            global_module_name (str):
                Global name of the module, parent packages are prepended and separated with ".".
            include_dir (str):
                Name of the main include dir.
            header (str|tuple):
                Name of the header file. Absolute paths or w.r.t. to include dir.
            runtime_linking (bool, optional):
                If runtime-linking code should be generated, defaults to False.
            util_pkg (str):
                Utility package that contains helper types and DLL loader routines.
            dll (str):
                Name of the DLL/shared object to link. Must not be none if
                `runtime_linking` is specified. Defaults to None.
            cflags (list(str), optional):
                Flags to pass to the C parser.
            module_opts (dict, optional):
                Per-module rendering options consumed inside ``render_python_interface_impl``.
                Keys: ``python_interface_always_return_tuple`` (bool, default False).
                The accumulator lists ``all`` and ``docstring_attributes`` are
                allocated per render pass — do not seed them here.
            \*\*opts:
                Further optional keyword arguments.
                See `CythonBackend` for further details.
        """
        global default_c_interface_decl_prolog
        global default_python_interface_decl_prolog
        self.module_opts = dict(module_opts) if module_opts else {}
        self.module_opts.setdefault(
            "python_interface_always_return_tuple", False
        )
        self.global_module_name = global_module_name

        parts = global_module_name.split(".")
        self.module_name = parts[-1]
        if len(parts) == 1:
            self.pkg_name = "."
        else:
            self.pkg_name = ".".join(global_module_name.split(".")[:-1])

        self.include_dir = include_dir
        self.header = header
        self.util_pkg = util_pkg
        self.runtime_linking = runtime_linking
        self.dll = dll
        self.cflags = cflags
        self.c_interface_decl_prolog = default_c_interface_decl_prolog
        self.c_interface_impl_prolog = default_c_interface_impl_prolog
        self.python_interface_decl_prolog = (
            default_python_interface_decl_prolog
        )
        self.python_interface_impl_prolog = (
            default_python_interface_impl_prolog
        )
        self.c_interface_decl_epilog = ""
        self.c_interface_impl_epilog = ""
        self.python_interface_decl_epilog = ""
        self.python_interface_impl_epilog = ""

        if isinstance(header, str):
            filename = header
            content = None
        elif isinstance(header, tuple):
            filename, content = header
        else:
            raise ValueError("type of 'headers' must be str or tuple")
        _log.info(" " + filename)
        if include_dir is not None:
            abspath = os.path.join(include_dir, filename)
        else:
            abspath = filename
        # libclang's unsaved_files keys must match the filename it tries to
        # open (i.e. the abspath), not the relpath. Use the abspath here so
        # in-memory rendered templates are actually served.
        unsaved_files = [(abspath, content)] if content is not None else None
        cflags = self.cflags + ["-I", f"{include_dir}"]
        parser = cparser.CParser(
            abspath, append_cflags=cflags, unsaved_files=unsaved_files
        )
        parser.parse()

        self.backend = CythonBackend.from_libclang_translation_unit(
            parser.translation_unit,
            filename,
            util_pkg,
            module_opts=self.module_opts,
            **opts,
        )

    def write_module_files(self, output_dir: str = None):
        """Write all files required to build this Cython/Python module.

        Args:
            module_name (str): Name of the module that should be generated. Influences filesnames.
        """
        module_name = self.module_name
        # C-level wrappers use the `cy` prefix (modern hip-python convention,
        # plan §B.5). Disambiguates Cython-level pxd/pyx files from anything
        # `c` might collide with.
        cy_module_name = f"cy{module_name}"

        python_interface_decl_prolog = (
            self.python_interface_decl_prolog
            + f"\ncimport {self.util_pkg}.types"
            + f"\ncimport {self.pkg_name}.{cy_module_name} as {cy_module_name}\n\n"
        )

        with open(
            f"{output_dir}/{cy_module_name}.pxd", "w", encoding="utf-8"
        ) as outfile:
            outfile.write(self.c_interface_decl_prolog)
            outfile.write(
                self.backend.render_c_interface_decl_part(
                    runtime_linking=self.runtime_linking,
                    module_name=module_name,
                )
            )
            outfile.write(self.c_interface_decl_epilog)
        with open(
            f"{output_dir}/{cy_module_name}.pyx", "w", encoding="utf-8"
        ) as outfile:
            outfile.write(self.c_interface_impl_prolog)
            outfile.write(
                self.backend.render_c_interface_impl_part(
                    util_pkg=self.util_pkg,
                    runtime_linking=self.runtime_linking,
                    dll=self.dll,
                    module_name=module_name,
                )
            )
            outfile.write(self.c_interface_impl_epilog)
        with open(
            f"{output_dir}/{module_name}.pxd", "w", encoding="utf-8"
        ) as outfile:
            outfile.write(python_interface_decl_prolog)
            outfile.write(
                self.backend.render_python_interface_decl_part(cy_module_name)
            )
            outfile.write(self.python_interface_decl_epilog)

        with open(
            f"{output_dir}/{module_name}.pyx", "w", encoding="utf-8"
        ) as outfile:
            (
                content,
                docstring_attributes,
            ) = self.backend.render_python_interface_impl_part(
                cy_module_name,
                module_name=module_name,
                runtime_linking=self.runtime_linking,
            )
            MODULE_DOCSTRING = (
                self.backend.root.render_python_docstring(cy_module_name)
                .lstrip('r"')
                .strip('"')
                + "\n"
            )
            if len(docstring_attributes):
                MODULE_DOCSTRING += "Attributes:\n" + textwrap.indent(
                    "\n".join(docstring_attributes), " " * 4
                )
            outfile.write(
                self.python_interface_impl_prolog.replace(
                    "[MODULE_DOCSTRING]", MODULE_DOCSTRING
                )
            )
            outfile.write(content)
            outfile.write(self.python_interface_impl_epilog)

        # Type-stub file for the high-level Python module. Used by
        # static type checkers (mypy, pyright) and IDEs to discover the
        # module's symbols without the compiled extension on sys.path.
        # The cy<name> C-level wrapper is cimport-only and deliberately
        # not stubbed (mapping void*/const char*/function-pointer typedefs
        # to fake Python type signatures would mislead).
        with open(
            f"{output_dir}/{module_name}.pyi", "w", encoding="utf-8"
        ) as outfile:
            outfile.write(self._render_pyi_stub(f"{cy_module_name}."))

    def _render_pyi_stub(self, cprefix: str) -> str:
        """Render a `.pyi` type-stub for the high-level Python module.

        Thin walker: each node knows how to render its own stub via
        `CythonMixin.render_pyi_stub` (overridden on `Function` to
        carry the real signature + docstring). This module-level
        method just iterates the same filtered node tree that drives
        the .pyx, concatenates the per-node stubs, and adds the
        module header + `__all__`.
        """
        lines = [
            "# AUTO-GENERATED by the hip-python code generator.",
            f"# Type stubs for {self.global_module_name}. Edits will be overwritten.",
            "",
            "import enum",
        ]
        if self.util_pkg:
            # The wrapper classes inherit `<util_pkg>.types.Pointer`.
            lines.append(f"import {self.util_pkg}.types")
        lines.extend(["from typing import Any", ""])
        names = []
        if self.runtime_linking:
            lines.append(
                "def has_symbol(name: str | bytes | bytearray) -> bool: ..."
            )
            lines.append("")
            names.append("has_symbol")
        for node in self.backend.walk_filtered_nodes():
            name = getattr(node, "name", None)
            # A leading underscore hides a compiler predefine (`__llvm__`,
            # `__GNUC__`), never a class: records such as `_hiprtcProgram`
            # are part of the public surface and the .pyx binds them.
            if not name or (
                name.startswith("_")
                and isinstance(node, _entities.MacroDefinition)
            ):
                continue
            # A hoisted anonymous record is `struct_0` to itself and
            # `<parent>_struct_0` to the module, and the number restarts
            # inside every parent. The global name is the one the .pyx
            # binds, so it is the only one worth stubbing.
            try:
                rendered_name = getattr(node, "cython_global_name", None) or (
                    node.renamer(name)
                    if callable(getattr(node, "renamer", None))
                    else name
                )
            except Exception:
                rendered_name = name
            # An anonymous enum has no name of its own -- libclang spells
            # it `enum (unnamed at <file>:<line>)` -- but the .pyx binds
            # its constants at module level, so it is stubbed all the same.
            anonymous_enum = (
                isinstance(node, _entities.Enum) and node.is_anonymous
            )
            if not anonymous_enum and (
                not rendered_name or not rendered_name.isidentifier()
            ):
                continue
            stub = node.render_pyi_stub(
                cprefix,
                override_name=rendered_name,
                module_opts=self.module_opts,
            )
            if not stub:
                continue
            lines.extend(stub)
            lines.append("")
            if anonymous_enum:
                names.extend(node.python_enum_constant_names)
            else:
                names.append(rendered_name)
        if names:
            lines.append("__all__ = [")
            for n in sorted(set(names)):
                lines.append(f"    {n!r},")
            lines.append("]")
        lines.append("")
        return "\n".join(lines)
