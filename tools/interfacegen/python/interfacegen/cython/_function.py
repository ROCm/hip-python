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


"""Cython backend — function submodule.

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

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults, _doxygen, _entities, _mixins
from ._defaults import *  # noqa: F401,F403
from ._doxygen import *  # noqa: F401,F403
from ._entities import *  # noqa: F401,F403
from ._mixins import *  # noqa: F401,F403

__all__ = [
    "Function",
]


class Function(tree.Function, CythonMixin, Typed):
    class SignatureMember:
        def __init__(self, value: str, typename: str, description: str):
            self.value: str = value
            self.typename: str = typename
            self.description: str = description

    def __init__(self, *args, **kwargs):
        tree.Function.__init__(self, *args, **kwargs)
        CythonMixin.__init__(self)
        self.modifiers_lazy_loader = ""
        self.error_return_value_lazy_loader = None
        self._python_return_values_to_prepend = []
        # Full hand-written overrides for the high-level Python interface.
        # When set (typically by a recipe ``node_init``), they let the
        # generator emit a verbatim ``def`` body / docstring for this one
        # function instead of the mechanical emitter output.
        # Both default to ``None`` (no override).
        self.python_interface_impl_override = None
        self.python_docstring_override = None

    def prepend_python_return_value(
        self, value: str, typename: str, description: str
    ):
        """Prepend a return value when rendering the Python interface of this node.

        Args:
            value (`str`):
                The value expression to prepend, typically a constant like 0 or an enum constant.
            typename (`str`):
                The typename of the return value that should appear in the docstring.
            description (`str`):
                The description of the return value that should appear in the docstring.
                Not implemented yet!
        """
        _log.debug(
            f"<{self.render_location()}> function {self.name}: prepend parm: {(value, typename, description)}"
        )
        self._python_return_values_to_prepend.append(
            Function.SignatureMember(value, typename, description)
        )

    # NOTE: `python_interface_always_return_tuple` was previously a
    # class-level attribute (default True) that recipes mutated globally.
    # It is now per-generator opts under
    # `module_opts["python_interface_always_return_tuple"]` (default False),
    # threaded as a kwarg into `render_python_interface_impl` /
    # `_render_python_docstring`.

    @property
    def has_python_body_prolog(self):
        """If any code has been prepended before the C interface call."""
        return hasattr(self, "_python_body_prolog")

    @property
    def has_python_body_epilog(self):
        """If any code has been prepended before the return statement."""
        return hasattr(self, "_python_body_epilog")

    def python_body_prepend_before_c_interface_call(self, code: str):
        """Prepend code right before the C interface call.

        Note:
            Additional call inserts code below the previously prepended code.
        """
        if not self.has_python_body_prolog:
            setattr(self, "_python_body_prolog", [])
        self._python_body_prolog.append(code)

    def python_body_prepend_before_return(self, code: str):
        """Prepend code right before the return statement.

        Note:
            Additional call inserts code below the previously prepended code.
        """
        if not self.has_python_body_epilog:
            setattr(self, "_python_body_epilog", [])
        self._python_body_epilog.append(code)

    @property
    def _has_funptr_parm(self):

        for node in self.walk():
            if isinstance(node, tree.Parm):
                if isinstance(node.typeref, tree.FunctionPointer):
                    return True
        return False

    def render_c_interface_decl(self, modifiers_front="", modifiers=""):

        typename = self.cython_global_typename
        name = self.cython_name
        parm_decls = ",".join([parm.cython_repr for parm in self.parms])
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{modifiers_front}{typename} {name}({parm_decls}){modifiers}
"""

    def render_cython_lazy_loader_decl(self):
        return self.render_c_interface_decl(
            modifiers_front="cdef ", modifiers=self.modifiers_lazy_loader
        )

    @property
    def cython_funptr_name(self):
        global c_interface_funptr_name_template
        return c_interface_funptr_name_template.format(name=self.cython_name)

    def render_cython_lazy_loader_def(self):

        funptr_name = self.cython_funptr_name

        parm_types = ",".join(
            [parm.cython_global_typename for parm in self.parms]
        )
        parm_names = ",".join(self.parm_names(self.renamer))
        typename = self.global_typename(
            self.sep, self.renamer, prefer_canonical=True
        )
        return f"""\
cdef void* {funptr_name} = NULL
{self.render_cython_lazy_loader_decl().strip()}:
    global {funptr_name}
    if __init_symbol(&{funptr_name},"{self.name}") > 0:
        {'return ' + self.error_return_value_lazy_loader if self.error_return_value_lazy_loader else 'pass'}
    {'' if self.is_void else 'return '}(<{typename} (*)({parm_types}) noexcept nogil> {funptr_name})({parm_names})
"""

    def _python_interface_retval_typename(self):
        """Returns a docstring expression for the return value type."""

        typename = self.cython_global_typename_no_const
        if self.is_void:
            return "None"
        elif self.is_basic_type or self.is_pointer_to_char(degree=1):
            return CYTHON_AUTOCONV_TO_PYTHON_TYPES(typename)
        elif self.is_enum or self.is_record:
            return typename
        else:
            return None

    # flake8: noqa: C901
    # TODO break function apart to reduce complexity
    def _render_python_docstring(
        self,
        out_arg_names: list,
        parm_python_types: dict,
        *,
        module_opts: dict = None,
    ):
        """Converts doxygen comment to a Python docstring using the doxyparser API.

        ``module_opts`` is the per-module rendering opts dict carried by
        the active CythonModuleGenerator. Used here to read
        ``python_interface_always_return_tuple``. When called outside of
        the main render pass (.pyi stub generation, standalone docstring
        rendering), pass None and the new default (False) applies.
        """
        # Verbatim override (set by a recipe node_init) wins: keeps the
        # .pyx body and the .pyi stub docstring in lockstep.
        if self.python_docstring_override is not None:
            return self.python_docstring_override
        _module_opts = module_opts if module_opts is not None else {}
        # TODO handle groups; issue detecting addgroup; detecting ingroup is easier

        doxyparsetree = self.doxygen_conv.parse_structure(
            self._raw_comment_cleaned()
        )
        sections = list(doxyparsetree.children)
        # brief
        docstring_body = self._render_doxygen_brief(
            sections,
            log_prefix=f"<{self.render_location()}> function {self.name}: ",
            missing_text="(No short description, might be part of a group.)",
            host_node=self if isinstance(self, tree.Node) else None,
        )

        # other sections, return values and parameters
        single_level_indent = " " * 4
        docstring_returns = []
        docstring_args = {}
        docstring_out_arg_returns = []
        parms_still_to_be_documented = [parm.name for parm in self.parms]
        in_inout_parm_names = [
            name
            for name in parms_still_to_be_documented
            if name not in out_arg_names
        ]

        for section in sections:
            if section.kind in (
                "result",
                "return",
                "returns",
            ):
                descr = self._render_doxygen_section_body(
                    section, outer_indent=single_level_indent
                ).lstrip("-* \t")
                docstring_returns.append(descr)
            elif section.kind == "retval":
                # \retval entries always describe the C return value (the
                # first tuple entry). Aggregate each one into the return
                # value description rather than rendering separate
                # "Retval:" sections (which would drop the value name).
                # tokens[1] is the return value name; it may have been
                # markdown-quoted in the header (`` `FOO` ``).
                retval_name = str(section.tokens[1]).strip("`")
                descr = (
                    self._render_doxygen_section_body(section, outer_indent="")
                    .strip()
                    .lstrip("-* \t")
                    .strip()
                )
                entry = self.to_sphinx_pyobj(retval_name)
                if descr:
                    entry += f": {descr}"
                docstring_returns.append(entry)
            elif section.kind == "param":
                # ['\\param', '[in]', 'param1', 'Description text is here.']
                names = section.tokens[2]
                # Args:
                #    <arg>: line1
                #       line2
                # ^ hence, 2x indent for descr
                descr = (
                    self._render_doxygen_section_body(
                        section, outer_indent=single_level_indent * 2
                    ).rstrip()
                    + "\n"
                )
                descr = descr.lstrip("-*")
                # example for tokens[1]: `[ in , out ]`
                dir = (
                    (
                        f" -- *{section.tokens[1][1:-1].replace(' ','').upper()}*"
                    )
                    if section.tokens[1] is not None
                    else ""
                )
                for name in names:
                    if not len(descr.strip()):
                        _log.warning(
                            f"<{self.render_location()}> function {self.name}: doxygen: doxygen param '{name}' has empty documentation."
                        )

                    if name in parms_still_to_be_documented:
                        type_info = "/".join(
                            [
                                CythonMixin.to_sphinx_pyobj(p)
                                for p in parm_python_types[name].split("/")
                            ]
                        )
                        parms_still_to_be_documented.remove(name)
                    else:
                        type_info = ""
                        _log.warning(
                            f"<{self.render_location()}> function {self.name}: doxygen: doxygen param '{name}' is not part of function signature."
                        )

                    if name in out_arg_names:
                        docstring_out_arg_returns.append(
                            f"{single_level_indent}{type_info}:\n{descr}"
                        )
                    else:
                        if len(type_info):
                            type_info = f" ({type_info})"
                        docstring_args[name] = (
                            name + type_info,
                            dir,
                            "\n" + descr,
                        )
            elif section.kind != "brief":
                docstring_body += self._render_doxygen_simple_section(
                    section, single_level_indent
                )

        # Combine multiple return statements
        if not len(docstring_returns) and not self.is_void:
            _log.warning(
                f"<{self.render_location()}> function {self.name}: doxygen: undocumented return value."
            )
        if len(docstring_returns) and self.is_void:
            _log.warning(
                f"<{self.render_location()}> function {self.name}: doxygen: has return section but is void."
            )
        combined_docstring_return = None
        if not self.is_void:
            # Always document the (status enum) return value when the function
            # is non-void, even if there is no `@return` section. The runtime
            # wrapper returns it as element 0 of the tuple, so it must be the
            # first documented return entry; otherwise it is silently dropped.
            retval_typename = self._python_interface_retval_typename()
            combined_docstring_return = (
                f"{CythonMixin.to_sphinx_pyobj(retval_typename)}"
            )
            if len(docstring_returns) > 1:
                bullets = []
                for e in docstring_returns:
                    e = textwrap.dedent(e).strip()
                    # Align continuation lines under the bullet text.
                    e = e.replace("\n", "\n  ")
                    bullets.append(f"- {e}")
                combined_docstring_return += ": One of:\n" + textwrap.indent(
                    "\n".join(bullets), single_level_indent * 2
                )
            elif len(docstring_returns) == 1:
                combined_docstring_return += ": " + docstring_returns[0]
            else:
                combined_docstring_return += ": (undocumented)"
            docstring_returns.clear()

        # Prepend user-prescribed return values
        for retval in self._python_return_values_to_prepend:
            docstring_returns.append(
                f"{single_level_indent}{self.to_sphinx_pyobj(retval.typename)}:\n{textwrap.indent(retval.description,single_level_indent*2)}"
            )
        if combined_docstring_return is not None:
            docstring_returns.append(combined_docstring_return)

        # Args
        # append undocumented arguments too but warn
        if len(parms_still_to_be_documented):
            for name in parms_still_to_be_documented:
                _log.warning(
                    f"<{self.render_location()}> function {self.name}: doxygen: function arg '{name}' is not documented."
                )
                type_info = "/".join(
                    [
                        CythonMixin.to_sphinx_pyobj(p)
                        for p in parm_python_types[name].split("/")
                    ]
                )
                type_info = f" ({type_info})"
                if name in out_arg_names:
                    docstring_out_arg_returns.append(
                        f"{single_level_indent}{name+type_info}:\n{single_level_indent}(undocumented)"
                    )
                else:
                    docstring_args[name] = (
                        name + type_info,
                        "",
                        f"\n{single_level_indent*2}(undocumented)\n",
                    )
        # now generate the arguments
        if len(docstring_args):
            docstring_body += "\nArgs:\n"

            for name in in_inout_parm_names:
                (head, dir, descr) = docstring_args[name]
                docstring_body += f"{single_level_indent}{head}{dir}:{descr}\n"

        # Add the additional out parameters as returns
        docstring_returns += docstring_out_arg_returns

        if len(docstring_returns):
            docstring_body += "\nReturns:\n"
            if len(docstring_returns) > 1 or _module_opts.get(
                "python_interface_always_return_tuple", False
            ):
                docstring_body += f"{single_level_indent}A {self.to_sphinx_pyobj('tuple')} of size {len(docstring_returns)} that contains (in that order):\n\n"
                prefix = "* "
            else:
                prefix = ""
            for descr in docstring_returns:
                docstring_body += (
                    textwrap.indent(
                        prefix + descr.lstrip(" \t\n*-"), single_level_indent
                    ).rstrip()
                    + "\n"
                )

        # Always-on: original C signature so the caller can cross-reference
        # the source header (especially when an unclassified pointer parm
        # falls back to ``Pointer`` + ``INOUT``).
        c_signature_block = self._render_c_signature_section(
            single_level_indent
        )
        if c_signature_block:
            docstring_body += c_signature_block

        # Clean result
        # remove multiple blank lines
        docstring_body = self.docstring_cleaner(docstring_body)
        # Mop up any doxygen tags the structured parser left behind.
        docstring_body = self._postprocess_leaked_doxygen_tags(docstring_body)
        # remove multiple blank lines
        docstring_body = re.sub(
            r"(\n\s*)+\n+", "\n\n", docstring_body
        ).rstrip()
        return f'r"""{docstring_body}\n"""'  # r required if verbatim/code is in body

    def _render_c_signature_section(self, single_level_indent: str) -> str:
        """Build a ``C signature`` docstring section showing the verbatim
        original C declaration: return type + function name + parm
        declarations as written in the source header.

        Emitted as an RST ``.. rubric::`` (lighter than a section heading
        so it doesn't pollute the page TOC) followed by a
        ``.. code-block:: c`` so Sphinx applies C syntax highlighting to
        the signature body. The rubric+code-block pair is left-anchored
        — the ``single_level_indent`` parameter is used for the code
        body inside the directive, matching the indentation Sphinx
        expects for directive content.

        Synthesized from the libclang cursor's type spellings rather than
        the original source extent — robust to ``unsaved_files`` parsing
        and avoids needing a separate source-text accessor.
        """
        try:
            ret_type = self.cursor.result_type.spelling
        except Exception:
            return ""
        parts = []
        for p in self.parms:
            try:
                ptype = p.cursor.type.spelling
                pname = p.cursor.spelling or ""
            except Exception:
                continue
            parts.append(f"{ptype} {pname}".rstrip())
        sig = f"{ret_type} {self.name}({', '.join(parts)})"
        return (
            f"\n\n.. rubric:: C signature\n\n"
            f".. code-block:: c\n\n"
            f"{single_level_indent}{sig}\n"
        )

    # `_add_module_cprefix` / `_PRIMITIVE_C_TYPES_FOR_CPREFIX` moved to
    # `CythonMixin` (shared by `Function` and `Field`); still reachable
    # here as `Function._add_module_cprefix(...)` via inheritance.

    @staticmethod
    def _parm_qualifiers(parm: Parm) -> str:
        result = ""
        if (
            parm.is_pointer_to_record(-1)
            or parm.is_pointer_to_enum(-1)
            or parm.is_pointer_to_basic_type(-1)
            or parm.is_pointer_to_void(-1)
        ):
            if parm.has_innermost_type_layer_const_modifier:
                result += "const "
        return result

    def _analyze_parms(self, cprefix: str):

        parm_python_types = (
            {}
        )  # Python type names of signature and out args, always use original typename as key
        sig_args = []  # argument definitions that appear in the signature
        out_args = []  # return values, might include conversions
        out_parms = (
            []
        )  # names of the return values, required for identifying doxygen parameters
        c_interface_call_args = (
            []
        )  # arguments that are passed to the C interface
        # Parallel list to c_interface_call_args. Each entry is either
        # None (the corresponding call_arg is inline in the cy* call) or
        # a complete `cdef <T> _cy_<f>__arg_N = <expr>` line that runs
        # before the call to materialize a Python-touching expression
        # (IntEnum .value lookup, fromPyobj factory call, Pointer
        # wrapper construction, etc.) into a typed C local. Both
        # emitters consume this list.
        c_interface_prehoist = []
        prolog = []  # additional code before the C interface call

        def _with_cprefix(c_type: str) -> str:
            return Function._add_module_cprefix(c_type, cprefix)

        def _append_call_arg(
            passthrough: str = None, *, hoist: CallArgHoist = None
        ):
            """Register one parameter passed to the cy* C call.

            Exactly one of ``passthrough`` / ``hoist`` must be given.

            ``passthrough`` form — the argument is a pure-C expression
            (typed parm ref, cdef-class ``_ptr`` field access,
            address-of, constantarray cast) that is GIL-safe inside
            ``with nogil:``. The expression is appended verbatim to the
            cy* call args and no prehoist line is emitted.

            ``hoist`` form — the argument touches Python (e.g.
            ``Wrapper.fromPyobj(arg).getPtr()`` or ``parm.value`` on
            an IntEnum). It is always rendered as a pre-call ``cdef <T>
            _cy_<func>__arg_N = <expr>`` line appended to
            ``c_interface_prehoist``, with the symbol appended to
            ``c_interface_call_args``.

            The rule is emitter-independent: anything derived from
            Python gets a named local, only pure-C expressions stay
            inline. Two reasons, either of which suffices:

            * Nothing that touches Python may run inside ``with
              nogil:``, so the with-nogil emitter needs the value
              materialized before the block.

            * A wrapper-bound hoist borrows memory owned by a wrapper
              temporary, and Cython releases that temporary before the
              cy* call runs — with the GIL held just as much as without
              it. Only the pre-call form (which binds the wrapper to an
              ``_obj`` local) keeps the memory alive across the call;
              see :class:`CallArgHoist`.
            """
            assert (passthrough is None) ^ (
                hoist is None
            ), "_append_call_arg: exactly one of passthrough / hoist required"
            if hoist is None:
                c_interface_prehoist.append(None)
                c_interface_call_args.append(passthrough)
            else:
                arg_idx = len(c_interface_call_args)
                arg_name = f"_cy_{self.cython_name}__arg_{arg_idx}"
                c_interface_prehoist.append(hoist.render_prehoist(arg_name))
                c_interface_call_args.append(arg_name)

        def handle_callee_allocated_ptr_parm(parm: Parm):
            # Handles OUT_CALLEE_ALLOCATED pointers only (dispatch keys on
            # `is_out_callee_allocated_ptr`, i.e. `intent.allocated_by_callee`):
            # the callee produces a fresh value / handle / string, which is
            # synthesized as a return-tuple entry. Caller-allocated OUT
            # buffers (plain OUT) take the caller-allocated path instead and
            # never reach here. On a shape this can't synthesize it raises
            # `CodegenUnsupportedPattern`; the dispatcher then degrades to
            # the caller-allocated path.
            nonlocal out_args
            nonlocal out_parms
            nonlocal c_interface_call_args
            nonlocal prolog
            nonlocal cprefix

            parm_name = parm.cython_name
            parm_innermost_type = parm.lookup_innermost_type()
            qualifiers = Function._parm_qualifiers(parm)

            out_parms.append(
                parm
            )  # append original name as we need to compare vs the documentation

            if (
                parm.is_pointer_to_basic_type(degree=1)
                and parm.actual_rank == 0
            ):
                # Take the base spelling from the same renderer that produced
                # the parameter's type in the .pxd. Reading the canonical
                # pointee's spelling off libclang instead would reintroduce the
                # host's data model here (`size_t *` declared in the .pxd
                # against an `unsigned long long` local on Windows, an
                # `unsigned long` one on Linux), and Cython rejects the
                # resulting pointer mismatch.
                parm_typename = parm.render_type(
                    parm.sep, parm.renamer, prefer_canonical=True
                ).base_typename
                prolog.append(f"cdef {parm_typename} {parm_name}")
                out_args.append(parm_name)  # TODO modify for char* pointer
                # Pure C: address-of a cdef-local typed scalar.
                _append_call_arg(f"&{parm_name}")
                parm_python_types[parm.name] = CYTHON_AUTOCONV_TO_PYTHON_TYPES(
                    parm_typename
                )
            elif parm.is_pointer_to_enum(degree=1):
                parm_typename = parm.lookup_innermost_type().cython_name
                prolog.append(f"cdef {cprefix}{parm_typename} {parm_name}")
                # Pure C: address-of a cdef-local enum scalar.
                _append_call_arg(f"&{parm_name}")
                out_args.append(
                    f"{parm_typename}({parm_name})"
                )  # conversion from c... type required
                parm_python_types[parm.name] = parm_typename
            elif parm.is_pointer_to_record(
                degree=2
            ) or parm.is_pointer_to_function_proto(degree=2):
                parm_typename = parm_innermost_type.cython_global_name
                # Type the wrapper local with cdef so the inline
                # `&{parm}._ptr` inside `with nogil:` is a typed C
                # field address (GIL-safe). Without cdef the local
                # is Python-typed and Cython rejects `&local._ptr`
                # as "Storing unsafe C derivative of temporary
                # Python reference".
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.fromPtr(NULL)"
                )
                _append_call_arg(
                    f"<{qualifiers}{cprefix}{parm_typename}**>&{parm_name}._ptr"
                )  # ! must be lvalue expression, can't use getElementPtr
                parm_python_types[parm.name] = parm_typename
                out_args.append(
                    f"None if {parm_name}._ptr == NULL else {parm_name}"
                )
            elif parm.is_pointer_to_record(degree=1):
                parm_typename = parm_innermost_type.cython_global_name
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.new()"
                )
                _append_call_arg(
                    f"<{qualifiers}{cprefix}{parm_typename}*>{parm_name}._ptr"
                )  # ! must be lvalue expression, can't use getElementPtr
                parm_python_types[parm.name] = parm_typename
                out_args.append(f"{parm_name}")
            elif parm.is_pointer_to_constantarray_of_basic_type(degree=2):
                if isinstance(
                    parm_innermost_type, tree.ConstantArray
                ):  # type has a wrapper class
                    parm_typename = parm_innermost_type.cython_name
                    cparm_typename = f"{cprefix}{parm_typename}**"
                else:
                    parm_typename = parm.ptr_complicated_type_handler(parm)
                    cparm_typename = parm.cursor.type.get_canonical().spelling
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.fromPtr(NULL)"
                )
                _append_call_arg(
                    f"<{cparm_typename}>&{parm_name}._ptr"
                )  # ! must be lvalue expression, can't use getElementPtr
                parm_python_types[parm.name] = parm_typename
                out_args.append(
                    f"None if {parm_name}._ptr == NULL else {parm_name}"
                )
            elif parm.is_pointer_to_constantarray_of_basic_type(degree=1):
                if isinstance(
                    parm_innermost_type, tree.ConstantArray
                ):  # type has a wrapper class
                    parm_typename = parm_innermost_type.cython_name
                    cparm_typename = f"{cprefix}{parm_typename}*"
                else:
                    parm_typename = parm.ptr_complicated_type_handler(parm)
                    cparm_typename = parm.cursor.type.get_canonical().spelling
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.new()"
                )
                _append_call_arg(
                    f"<{cparm_typename}*>{parm_name}._ptr"
                )  # ! must be lvalue expression, can't use getElementPtr
                parm_python_types[parm.name] = parm_typename
                out_args.append(f"{parm_name}")
            elif parm.is_pointer_to_basic_type(
                degree=-2
            ) or parm.is_pointer_to_void(degree=-2):
                parm_typename = parm.ptr_complicated_type_handler(parm)
                cparm_typename = parm.cursor.type.get_canonical().spelling
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.fromPtr(NULL)"
                )
                _append_call_arg(
                    f"\n{indent*2}<{cparm_typename}>&{parm_name}._ptr"
                )
                parm_python_types[parm.name] = f"{parm_typename}/object"
                out_args.append(
                    f"None if {parm_name}._ptr == NULL else {parm_name}"
                )
            elif parm.is_pointer_to_basic_type(
                degree=-1
            ) or parm.is_pointer_to_void(degree=-1):
                parm_typename = parm.ptr_complicated_type_handler(parm)
                cparm_typename = parm.cursor.type.get_canonical().spelling
                prolog.append(
                    f"cdef {parm_typename} {parm_name} = "
                    f"{parm_typename}.fromPtr(NULL)"
                )
                _append_call_arg(
                    f"\n{indent*2}<{cparm_typename}>{parm_name}._ptr"
                )
                parm_python_types[parm.name] = f"{parm_typename}/object"
                out_args.append(
                    f"None if {parm_name}._ptr == NULL else {parm_name}"
                )
            else:
                # If the argument was not removed from the parameter list,
                # we did not add an additional return value.
                # Hence, we remove the previously added original
                # name (see top of routine) from the out_arg_names list.
                canonical = parm.cursor.type.get_canonical().spelling
                _log.error(
                    f"<{self.render_location()}> function {self.name}: parm {parm_name}: not handled, canonical C type: '{canonical}'"
                )
                raise CodegenUnsupportedPattern(
                    self.name, parm_name, canonical
                )

        def emit_datahandle_(
            parm_typename: str, parm: tree.Parm, cprefix: str = ""
        ):
            global indent
            nonlocal sig_args
            nonlocal c_interface_call_args
            nonlocal parm_python_types

            # TODO hacky and does not consider volatile e.g.
            if parm_typename.startswith("const "):
                cprefix = "const " + cprefix
                parm_typename = parm_typename[len("const ") :]

            parm_name = parm.cython_name
            handler_name = parm.ptr_complicated_type_handler(parm)
            sig_args.append(f"object {parm_name}")
            # Touches Python via .fromPyobj() (a cdef staticmethod that
            # is not nogil-callable). Hoist into a typed C local so
            # the cy* call inside `with nogil:` only sees the C value.
            #
            # Use ``.getPtr()`` (a cdef method returning ``void*``)
            # rather than ``._ptr`` (a cdef field). Storing the
            # value of a method-call result is safe; storing
            # ``temp._ptr`` would be flagged by Cython as "Storing
            # unsafe C derivative of temporary Python reference"
            # because the field reference is tied to the temporary
            # wrapper object's lifetime, while the method-call
            # return value is a copied-by-value C scalar.
            #
            # The local ``cprefix`` here already encodes whether a
            # cy* module prefix is required (caller passes "" for
            # primitive-typed parms like ``const char *``, the cy*
            # module name like "cyhip." for cy*-typed parms) and any
            # ``const`` qualifier was just folded in above. Use it
            # directly for the hoist type — _with_cprefix would
            # blindly add the outer cy* prefix even to primitive
            # types.
            _append_call_arg(
                hoist=CallArgHoist(
                    c_type=f"{cprefix}{parm_typename}",
                    wrapper_class=handler_name,
                    wrapper_factory=f"{handler_name}.fromPyobj({parm_name})",
                    pointer_extract="getPtr()",
                    cast_open=f"<{cprefix}{parm_typename}>",
                )
            )
            parm_python_types[parm.name] = f"{handler_name}/object"

        def emit_data_handle_for_ptr_to_void_basic_enum_(
            parm: tree.Parm, cprefix: str
        ):
            parm_typename = (
                parm.cython_global_typename
                if parm.has_typeref
                else parm.renamer(parm.cursor.type.get_canonical().spelling)
            )
            emit_datahandle_(
                parm_typename,
                parm,
                cprefix=(
                    ""
                    if parm.is_innermost_canonical_type_layer_of_basic_type_or_void
                    else cprefix
                ),
            )

        def handle_caller_allocated_ptr_(parm: Parm):
            # Handles IN and INOUT pointers — both caller-allocated: the
            # caller owns the buffer and the callee reads (IN) and/or
            # fills (INOUT) it.
            global indent
            nonlocal c_interface_call_args
            nonlocal sig_args
            nonlocal cprefix

            parm_name = parm.cython_name
            parm_innermost_type = parm.lookup_innermost_type()

            if parm.is_pointer_to_record(
                degree=1, incomplete_array=True
            ) or parm.is_pointer_to_function_proto(
                degree=1, incomplete_array=True
            ):
                # The default rendering uses a per-type wrapper class
                # named after the innermost record (e.g. `hipblasContext`
                # for `hipblasHandle_t *`). That requires the wrapper
                # class to actually be emitted in this binding, which
                # in turn requires the recipe filter to admit the
                # innermost record. For foreign-prefix records like
                # libc `FILE` / `_IO_FILE` the filter rejects them and
                # the wrapper class is never defined — Cython would
                # fail with `undeclared name not builtin: _IO_FILE`.
                # Fall back to the handler-driven generic wrapper (same
                # path that the void-pointer branch below uses), which
                # defaults to `rocm.bindings.util.types.Pointer`. The
                # recipe can override per-binding via
                # `ptr_complicated_type_handler`.
                c_type = _with_cprefix(parm.cython_global_typename_no_const)
                node_filter = getattr(parm, "node_filter", None)
                if (
                    node_filter is None
                    or parm_innermost_type is None
                    or node_filter(parm_innermost_type)
                ):
                    # Per-type wrapper IS emitted — `getElementPtr()`
                    # already returns the typed pointer, no cast.
                    parm_typename = parm_innermost_type.cython_global_name
                    pointer_extract = "getElementPtr()"
                    cast_open = ""
                else:
                    # Foreign record — fall back to the generic
                    # Pointer wrapper. `getPtr()` returns `void *`, so
                    # the call needs an explicit cast to the cy*
                    # parameter type.
                    parm_typename = parm.ptr_complicated_type_handler(parm)
                    pointer_extract = "getPtr()"
                    cast_open = f"<{c_type}>"
                sig_args.append(f"object {parm_name}")
                parm_python_types[parm.name] = (
                    f"{parm_typename}/object"  # use original name as key
                )
                # Touches Python via .fromPyobj(). Hoist the resolved
                # C pointer into a typed local so the cy* call inside
                # `with nogil:` only sees a pure C pointer. Use the
                # cprefix-prefixed cy* type so Cython sees the C
                # struct / typedef from the cy* module rather than
                # the same-named Python wrapper class in this file.
                _append_call_arg(
                    hoist=CallArgHoist(
                        c_type=c_type,
                        wrapper_class=parm_typename,
                        wrapper_factory=f"{parm_typename}.fromPyobj({parm_name})",
                        pointer_extract=pointer_extract,
                        cast_open=cast_open,
                    )
                )
            elif parm.is_pointer_to_constantarray_of_basic_type(
                degree=1, incomplete_array=True
            ):
                if isinstance(
                    parm_innermost_type, tree.ConstantArray
                ):  # type has a wrapper class
                    parm_typename = (
                        parm_innermost_type.cython_name
                    )  # use cython name to get typedef name
                    sig_args.append(f"object {parm_name}")
                    parm_python_types[parm.name] = (
                        f"{parm_typename}/object"  # use original name as key
                    )
                    # Same hoist rationale as above.
                    _append_call_arg(
                        hoist=CallArgHoist(
                            c_type=_with_cprefix(
                                parm.cython_global_typename_no_const
                            ),
                            wrapper_class=parm_typename,
                            wrapper_factory=f"{parm_typename}.fromPyobj({parm_name})",
                            pointer_extract="getElementPtr()",
                        )
                    )
                else:  # type has no wrapper class, emit default handler
                    parm_typename = parm.cython_global_typename
                    emit_datahandle_(parm_typename, parm, "")
            elif parm.is_pointer_to_record(
                degree=-2, incomplete_array=True
            ) or parm.is_pointer_to_function_proto(
                degree=-2, incomplete_array=True
            ):
                # TODO: split and use rank == 0 (scalar) information to handle some record arg destroy funs such as 'hiprtcDestroyProgram(struct _hiprtcProgram**)'
                parm_typename = parm.cython_global_typename
                emit_datahandle_(parm_typename, parm, cprefix)
            elif parm.is_pointer_to_constantarray_of_basic_type(
                degree=-2, incomplete_array=True
            ):
                if isinstance(
                    parm_innermost_type, tree.ConstantArray
                ):  # type has a wrapper class
                    degree = parm.get_pointer_degree(incomplete_array=True)
                    parm_typename = (
                        parm_innermost_type.cython_name
                    )  # use cython name to get typedef name
                    emit_datahandle_(
                        parm_typename + "*" * degree, parm, cprefix
                    )
                else:  # type has no wrapper class, emit default handler
                    parm_typename = parm.cython_global_typename
                    emit_datahandle_(parm_typename, parm, "")
            elif (
                parm.is_pointer_to_void(degree=-1, incomplete_array=True)
                or parm.is_pointer_to_basic_type(
                    degree=-1, incomplete_array=True
                )
                or parm.is_pointer_to_enum(degree=-1, incomplete_array=True)
            ):
                emit_data_handle_for_ptr_to_void_basic_enum_(parm, cprefix)
            else:
                # Worst case: a pointer shape none of the structured
                # branches above bind (e.g. a caller-allocated OUT
                # `T *const **` / `T *const[]`). The caller owns the
                # buffer, so bind it as the generic Pointer wrapper
                # (`ptr_complicated_type_handler` defaults to
                # `rocm.bindings.util.types.Pointer`) and let the user
                # wire it via Pointer / ctypes. Never crash. This mirrors
                # the foreign-record fallback above. A recipe can override
                # the wrapper per-binding via `ptr_complicated_type_handler`.
                c_type = _with_cprefix(parm.cython_global_typename_no_const)
                parm_typename = parm.ptr_complicated_type_handler(parm)
                _log.warning(
                    f"<{self.render_location()}> function {self.name}: "
                    f"parm {parm.name}: no structured caller-allocated "
                    f"binding for canonical type "
                    f"{parm.cursor.type.get_canonical().spelling!r}; "
                    f"falling back to generic {parm_typename} wrapper"
                )
                sig_args.append(f"object {parm_name}")
                parm_python_types[parm.name] = f"{parm_typename}/object"
                _append_call_arg(
                    hoist=CallArgHoist(
                        c_type=c_type,
                        wrapper_class=parm_typename,
                        wrapper_factory=f"{parm_typename}.fromPyobj({parm_name})",
                        pointer_extract="getPtr()",
                        cast_open=f"<{c_type}>",
                    )
                )

        def handle_value_parm_(parm: Parm):
            if parm.is_autoconverted_by_cython:
                # Pure C: typed scalar parm passed by value.
                _append_call_arg(f"{parm_name}")
                sig_args.append(parm.cython_repr)
                parm_python_types[parm.name] = "/".join(
                    CYTHON_AUTOCONV_FROM_PYTHON_TYPES(
                        parm.cython_global_typename
                    )
                )  # use original name as key
            elif (
                parm.is_enum
            ):  # enums are not modelled as cdef class, so we cannot specify them as type
                parm_base_class_name = (
                    parm.lookup_innermost_type().python_base_class_name
                )
                sig_args.append(f"object {parm_name}")
                prolog.append(
                    textwrap.dedent(
                        f"""\
                    if not isinstance({parm_name},{parm_base_class_name}):
                        raise TypeError("argument '{parm_name}' must be of type '{parm_base_class_name}'")"""
                    )
                )
                # IntEnum .value is Python attribute access — must
                # happen with the GIL held. Hoist to a typed C enum
                # local before `with nogil:`. Use the cprefix-
                # prefixed cy* enum type so Cython resolves to the
                # C enum (not the same-named IntEnum class in this
                # file). No wrapper temporary involved (the IntEnum
                # is already bound to the function's `parm_name`
                # local), so a plain hoist suffices.
                _append_call_arg(
                    hoist=CallArgHoist(
                        c_type=_with_cprefix(
                            parm.cython_global_typename_no_const
                        ),
                        plain_expr=f"{parm_name}.value",
                    )
                )
                # Use the no_const spelling so the rendered docstring's
                # `:py:obj:` reference doesn't carry a leading `const`
                # (the Python-facing type doesn't have C cv-qualifiers
                # — `const hiptensorWorksizePreference_t` would format
                # as a broken `:py:obj:\`.const ...\`` link).
                parm_python_types[parm.name] = (
                    parm.cython_global_typename_no_const
                )
            elif parm.is_record or parm.is_basic_type_constantarray():
                parm_typename = parm.lookup_innermost_type().cython_name
                sig_args.append(f"object {parm_name}")
                # Touches Python via .fromPyobj() and a Python-typed
                # method chain. Hoist the dereferenced record value
                # into a typed C local (record is copied by value).
                # Use the cprefix-prefixed cy* struct type so the cdef
                # local names the C struct, not the same-named Python
                # wrapper class in this file.
                _append_call_arg(
                    hoist=CallArgHoist(
                        c_type=_with_cprefix(
                            parm.cython_global_typename_no_const
                        ),
                        wrapper_class=parm_typename,
                        wrapper_factory=f"{parm_typename}.fromPyobj({parm_name})",
                        pointer_extract="getElementPtr()[0]",
                    )
                )
                parm_python_types[parm.name] = parm_typename
            else:
                assert False, "should not be entered"

        for parm in self.parms:
            parm_name = parm.cython_name
            assert isinstance(parm, Parm)
            if parm.is_ptr:
                if parm.is_out_callee_allocated_ptr:
                    # OUT_CALLEE_ALLOCATED is the only intent that adds a
                    # return-tuple entry: the callee produces a fresh
                    # handle / scalar / string and we synthesize it as a
                    # return value. Snapshot mutable state so we can roll
                    # back if that synthesis can't build this parm's shape
                    # (e.g. `T *const **` / `T *const[]` — middle-const
                    # pointer shapes the codegen doesn't yet support). On
                    # an unsupported shape, degrade to the caller-allocated
                    # path: drop the synthesized entry and re-bind as a
                    # plain OUT pointer, so the return tuple is unchanged
                    # and the caller passes a Pointer / complicated type.
                    _log.debug(
                        f"<{self.render_location()}> function {self.name}: parm {parm.name}: classified as OUT-PTR (callee-allocated)"
                    )
                    _snap_out_parms = len(out_parms)
                    _snap_out_args = len(out_args)
                    _snap_call_args = len(c_interface_call_args)
                    _snap_prolog = len(prolog)
                    try:
                        handle_callee_allocated_ptr_parm(parm)
                    except CodegenUnsupportedPattern as exc:
                        del out_parms[_snap_out_parms:]
                        del out_args[_snap_out_args:]
                        del c_interface_call_args[_snap_call_args:]
                        del prolog[_snap_prolog:]
                        parm_python_types.pop(parm.name, None)
                        _log.warning(
                            f"<{self.render_location()}> function {self.name}: "
                            f"parm {parm.name}: callee-allocated OUT codepath "
                            f"unsupported for canonical type "
                            f"{exc.canonical_type!r}; degrading to "
                            f"caller-allocated (plain OUT) binding"
                        )
                        handle_caller_allocated_ptr_(parm)
                else:
                    # IN, plain (caller-allocated) OUT, and INOUT all share
                    # the caller-allocated path: the caller owns the buffer
                    # and we never add a return-tuple entry. This handler is
                    # total — unsupported shapes fall back to the generic
                    # Pointer wrapper rather than crashing.
                    _log.debug(
                        f"<{self.render_location()}> function {self.name}: parm {parm.name}: classified as IN/OUT/INOUT-PTR (caller-allocated)"
                    )
                    handle_caller_allocated_ptr_(parm)
            else:  # no ptr
                _log.debug(
                    f"<{self.render_location()}> function {self.name}: parm {parm.name}: classified as IN-VALUE"
                )
                handle_value_parm_(parm)

        fully_specified = len(list(self.parms)) == len(c_interface_call_args)
        if not fully_specified:
            _log.warning(
                f"interfacegen.cython: not all parameters could be classified for function {self.name} (from <{self.render_location()}>)"
            )
        setattr(self, "is_python_code_complete", fully_specified)
        assert len(parm_python_types) == len(
            c_interface_call_args
        ), f"{self.name=} {str(parm_python_types)=}"
        assert len(c_interface_prehoist) == len(
            c_interface_call_args
        ), f"{self.name=} prehoist/call_args length mismatch"

        return (
            fully_specified,
            sig_args,
            out_args,
            out_parms,
            c_interface_call_args,
            c_interface_prehoist,
            prolog,
            parm_python_types,
        )

    @property
    def _python_interface_retval(self):
        global python_interface_retval_template
        return python_interface_retval_template.format(name=self.cython_name)

    @property
    def _cy_python_interface_retval(self):
        """C-level return-value holder used by the `with nogil:` emitter.

        Distinct from `_python_interface_retval` (the
        `_<func>__retval` Python-side name used by the with-gil
        emitter) so the with-nogil function body can hold both the
        cdef C scalar (assigned inside the nogil block) and the
        wrapped Python value (inlined in the return tuple).
        """
        return f"_cy_{self.cython_name}__retval"

    def _render_python_interface_c_interface_call(
        self, cprefix: str, call_args: list, prehoist: list, out_args: list
    ):
        """Dispatcher: pick the with-nogil or with-gil emitter based on
        whether the cy* function declaration carries the ``nogil``
        modifier.

        Both modes are valid first-class options. ROCm recipes
        currently set ``modifiers_lazy_loader`` to include ``nogil``
        (see ``generators_hip.py:106`` and the other per-library
        generators), so the with-nogil emitter runs in production. The
        with-gil emitter handles recipes that intentionally keep the
        GIL across the C call (e.g. when the cy* function callbacks
        into Python).
        """
        if "nogil" in (self.modifiers_lazy_loader or ""):
            return self._render_python_interface_c_interface_call_with_nogil(
                cprefix,
                call_args,
                prehoist,
                out_args,
            )
        return self._render_python_interface_c_interface_call_with_gil(
            cprefix,
            call_args,
            prehoist,
            out_args,
        )

    def _render_python_interface_c_interface_call_with_gil(
        self, cprefix: str, call_args: list, prehoist: list, out_args: list
    ):
        """With-GIL emission: cy* call + inline Python wrap, preceded by
        the pre-call argument bindings.

        Used when the cy* function declaration is not
        ``nogil``-callable — every expression runs with the GIL held
        throughout, so the call/wrap can be inlined into the return
        tuple. This mode is appropriate when the cy* call itself may
        touch Python (e.g. via a callback) and therefore must hold the
        GIL.

        Holding the GIL does not extend the life of an argument's
        wrapper temporary, so this emitter uses the same pre-call
        ``cdef`` lines as the with-nogil one (see
        :class:`CallArgHoist`); they are emitted here ahead of the
        call. Only the retval wrap differs: it stays inline in the call
        expression, since the GIL is held there.
        """
        lines = [entry for entry in prehoist if entry is not None]
        typename = self.cython_global_typename
        retvalname = self._python_interface_retval
        retvalname_or_none = (
            f"None if {retvalname}._ptr == NULL else {retvalname}"
        )
        comma = ","
        c_interface_call = (
            f"{cprefix}{self.cython_name}({comma.join(call_args)})"
        )

        if self.is_void:
            lines.append(c_interface_call)
        elif self.is_basic_type:
            out_args.insert(0, retvalname)
            lines.append(f"cdef {typename} {retvalname} = {c_interface_call}")
        elif self.is_enum:
            out_args.insert(0, retvalname)
            lines.append(f"{retvalname} = {typename}({c_interface_call})")
        elif self.is_record:
            out_args.insert(0, retvalname)
            innermost_typename = (
                self.lookup_innermost_type().cython_global_name
            )
            # Using the innermost type ensures that the return value handler is a cdef class and not a Python object
            # that was inserted because of a typedef.
            lines.append(
                f"{retvalname} = {innermost_typename}.fromValue({c_interface_call})"
            )
        elif self.is_pointer_to_record():
            out_args.insert(0, retvalname_or_none)
            innermost_typename = (
                self.lookup_innermost_type().cython_global_name
            )
            # Using the innermost type ensures that the return value handler is a cdef class and not a Python object
            # that was inserted because of a typedef.
            lines.append(
                f"{retvalname} = {innermost_typename}.fromPtr({c_interface_call})"
            )
        elif self.is_pointer_to_char(
            degree=1
        ):  # TODO adapt to use ptr complicated type handler, result might be buffer
            out_args.insert(0, retvalname_or_none)
            handler = self.ptr_complicated_type_handler(self)
            lines.append(
                f"{retvalname} = {handler}.fromPtr(<void*>{c_interface_call})"
            )
        elif self.is_any_pointer:
            out_args.insert(0, retvalname_or_none)
            lines.append(
                f"{retvalname} = {self.util_types_prefix}Pointer.fromPtr(<void*>{c_interface_call})"
            )
        else:
            msg = "<{self.render_location()}> function {self.name}: return value type could not be classified."
            _log.warning(msg)
            raise RuntimeError(msg)
        return "\n".join(lines)

    def _format_retval_wrap(self, cy_retval: str) -> str:
        """Return the inline expression that wraps the C-level
        ``cy_retval`` for placement into the return tuple.

        Dispatches on the same seven retval-shape predicates as the
        with-gil emitter, but always produces an *expression*
        (no assignment) suitable for embedding directly into
        ``out_args``. Wraps that touch Python's type machinery
        (``IntEnum(...)``, ``T.fromValue(...)``, ``T.fromPtr(...)``)
        live here precisely because they cannot run inside
        ``with nogil:`` — they execute post-block.

        Caller is responsible for handling void retvals (this method
        is not invoked then).
        """
        if self.is_basic_type:
            # The C value is already the Python value — no wrap needed.
            return cy_retval
        if self.is_enum:
            # IntEnum.__call__ goes through Python — must be post-block.
            return f"{self.cython_global_typename}({cy_retval})"
        if self.is_record:
            innermost_typename = (
                self.lookup_innermost_type().cython_global_name
            )
            # T.fromValue is a Python wrapper construction — needs GIL.
            return f"{innermost_typename}.fromValue({cy_retval})"
        if self.is_pointer_to_record():
            innermost_typename = (
                self.lookup_innermost_type().cython_global_name
            )
            return (
                f"None if {cy_retval} == NULL else "
                f"{innermost_typename}.fromPtr({cy_retval})"
            )
        if self.is_pointer_to_char(degree=1):
            handler = self.ptr_complicated_type_handler(self)
            return (
                f"None if {cy_retval} == NULL else "
                f"{handler}.fromPtr(<void*>{cy_retval})"
            )
        if self.is_any_pointer:
            return (
                f"None if {cy_retval} == NULL else "
                f"{self.util_types_prefix}Pointer.fromPtr(<void*>{cy_retval})"
            )
        msg = (
            f"<{self.render_location()}> function {self.name}: "
            "return value type could not be classified."
        )
        _log.warning(msg)
        raise RuntimeError(msg)

    def _render_python_interface_c_interface_call_with_nogil(
        self, cprefix: str, call_args: list, prehoist: list, out_args: list
    ):
        """Multi-line emission that releases the GIL across the cy*
        call. Pre-block hoists materialize Python-touching arg
        expressions into typed C locals; the cy* call runs inside
        ``with nogil:``; the Python retval wrap is inlined post-block
        directly into ``out_args``.

        Returned string is unindented; the caller (``render_python_
        interface_impl``) applies the function-body indent uniformly.
        """
        lines = []

        # 1. Pre-block hoists. Each non-None entry is a complete
        #    `cdef T _cy_<f>__arg_N = <expr>` line (GIL held).
        for entry in prehoist:
            if entry is not None:
                lines.append(entry)

        cy_retval = self._cy_python_interface_retval

        # 2. cdef the C-level retval holder (skipped for void).
        # The bare ``cython_global_typename`` collides with same-named
        # Python wrappers in the high-level ``<module>.pyx``
        # (``hipError_t``, ``hiprtcResult`` are IntEnum classes,
        # ``dim3`` is a cdef-class wrapper, …). Add the cy* module
        # cprefix (skipping primitives like ``int`` / ``void``) so
        # the cdef declaration unambiguously names the C type.
        if not self.is_void:
            # A const *value* return has to lose the qualifier: Cython
            # rejects the assignment into `cdef const T var` ("Assignment
            # to const") even though the local only ever receives the
            # value, and const-correctness on the C call boundary is
            # already enforced by the cy* declaration. A pointer return
            # keeps it -- there the const belongs to the pointee, so the
            # local stays assignable, and dropping it makes Cython warn
            # that the assignment discards the qualifier.
            retval_typename = (
                self.cython_global_typename
                if self.is_any_pointer
                else self.cython_global_typename_no_const
            )
            retval_c_type = Function._add_module_cprefix(
                retval_typename,
                cprefix,
            )
            lines.append(f"cdef {retval_c_type} {cy_retval}")

        # 3. The `with nogil:` block — single statement (the cy* call).
        comma = ","
        cy_call = f"{cprefix}{self.cython_name}({comma.join(call_args)})"
        lines.append("with nogil:")
        if self.is_void:
            lines.append(f"{indent}{cy_call}")
        else:
            lines.append(f"{indent}{cy_retval} = {cy_call}")

        # 4. Post-block: inline the Python wrap (or `None if NULL else
        #    wrap`) directly into out_args. No `_py_<func>__retval`
        #    intermediate — the wrap appears once and only feeds the
        #    return tuple.
        if not self.is_void:
            out_args.insert(0, self._format_retval_wrap(cy_retval))

        return "\n".join(lines)

    def render_python_docstring(self, cprefix: str) -> str:
        """Public API for generating only the docstring."""
        (
            __fully_specified,
            __sig_args,
            __out_args,
            out_parms,  # required for parsing parameter documentation
            __call_args,
            __prehoist,
            __prolog,
            parm_python_types,
        ) = self._analyze_parms(cprefix)
        # TODO insert additional args here
        return (
            self._render_python_docstring(
                [p.name for p in out_parms], parm_python_types
            ),
            indent,
        )

    def render_python_interface_impl(
        self, cprefix: str, *, module_opts: dict
    ) -> str:
        """Public API for generating the full Python interface."""
        # Verbatim body override (set by a recipe node_init): emit the
        # hand-written ``def`` as-is, but still register the symbol in
        # ``__all__`` so the module surface is unchanged.
        if self.python_interface_impl_override is not None:
            module_opts["all"].append(self.cython_global_name)
            return self.python_interface_impl_override

        (
            fully_specified,
            sig_args,
            out_args,
            out_parms,  # required for parsing parameter documentation
            call_args,
            prehoist,
            prolog,
            parm_python_types,
        ) = self._analyze_parms(cprefix)
        result = "@cython.embedsignature(True)\n"
        result += (
            f"def {self.cython_name}({', '.join(sig_args)}):\n"
            + textwrap.indent(
                self._render_python_docstring(
                    [p.name for p in out_parms],
                    parm_python_types,
                    module_opts=module_opts,
                ),
                indent,
            ).rstrip()
            + "\n"
        )
        if self.has_python_body_prolog:
            prolog += self._python_body_prolog
        epilog = []
        if self.has_python_body_epilog:
            epilog += self._python_body_epilog
        if len(prolog):
            result += (
                textwrap.indent("\n".join(prolog), indent).rstrip() + "\n"
            )
        if fully_specified:
            # The with-nogil emitter returns multi-line text; the
            # with-gil emitter returns a single line. Indent uniformly
            # via textwrap.indent so both shapes land correctly inside
            # the function body.
            emission = self._render_python_interface_c_interface_call(
                cprefix,
                call_args,
                prehoist,
                out_args,
            )
            result += textwrap.indent(emission, indent).rstrip() + "\n"
            if len(epilog):
                result += (
                    textwrap.indent("\n".join(epilog), indent).rstrip() + "\n"
                )
            # prepend user-prescribed values
            out_args = [
                m.value for m in self._python_return_values_to_prepend
            ] + out_args
            if len(out_args) > 1:
                comma = ","
                result += f"{indent}return ({comma.join(out_args)})\n"
            elif len(out_args):
                if module_opts.get(
                    "python_interface_always_return_tuple", False
                ):
                    result += f"{indent}return ({out_args[0]},)\n"
                else:
                    result += f"{indent}return {out_args[0]}\n"
        else:
            _log.warning(
                f" function {self.cython_global_name}: not all parameters could be mapped"
            )
            result += f"{indent}pass"
        module_opts["all"].append(self.cython_global_name)
        return result

    # ------------------------------------------------------------------
    # .pyi rendering
    # ------------------------------------------------------------------

    def render_pyi_stub(
        self,
        cprefix: str,
        *,
        override_name: str = None,
        base: str = None,
        module_opts: dict = None,
    ):
        """Function override of `CythonMixin.render_pyi_stub`.

        Reuses `_analyze_parms` to recover the real parameter names,
        and `_render_python_docstring` to embed the same Args/Returns
        docstring the .pyx carries. Argument and return-value type
        annotations are intentionally bare — `parm_python_types` is
        union-valued (e.g. ``DeviceArray/object``) and would not
        round-trip cleanly to a Python annotation; the docstring is
        the source of truth for types.

        `base` is unused for functions (kept in the signature for
        a uniform call site with class-stub renderers). On any
        analysis exception, falls back to ``def name(*args, **kwargs)``.
        """
        del base  # functions don't have a base class
        name = override_name or self.cython_name
        if not name or not name.isidentifier():
            return None
        try:
            (
                _fully_specified,
                sig_args,
                _out_args,
                out_parms,
                _call_args,
                _prehoist,
                _prolog,
                parm_python_types,
            ) = self._analyze_parms(cprefix)
        except Exception:
            return [f"def {name}(*args, **kwargs): ..."]
        py_params = []
        for sa in sig_args:
            _ctype, pname, default = _pyi_split_sig_arg(sa)
            if default is not None:
                py_params.append(f"{pname}={default}")
            else:
                py_params.append(pname)
        try:
            docstring = self._render_python_docstring(
                [p.name for p in out_parms],
                parm_python_types,
                module_opts=module_opts,
            )
        except Exception:
            docstring = None
        sig = ", ".join(py_params)
        lines = [f"def {name}({sig}):"]
        if docstring:
            for dl in docstring.splitlines():
                lines.append(f"    {dl}" if dl else "")
        lines.append("    ...")
        return lines


def _pyi_split_sig_arg(sig_arg: str):
    """Parse a Cython sig_arg like 'unsigned long size' or 'object foo'
    into (cython_type, name, default). Used by `Function.render_pyi_stub`."""
    if "=" in sig_arg:
        decl, default = sig_arg.split("=", 1)
        decl = decl.rstrip()
        default = default.strip()
    else:
        decl = sig_arg
        default = None
    parts = decl.rsplit(" ", 1)
    if len(parts) == 1:
        ctype = None
        name = parts[0]
    else:
        ctype, name = parts
    name = name.lstrip("*")
    return (ctype.strip() if ctype else None, name.strip(), default)
