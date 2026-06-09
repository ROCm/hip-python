# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import logging
import os
import textwrap

_log = logging.getLogger("interfacegen")

python_interface_pyobj_role_template = r":py:obj:`~.{name}`"

from interfacegen.cython import (
    CythonModuleGenerator,
)
from interfacegen.tree import (
    AnonymousEnum,
    Enum,
    Function,
    FunctionPointer,
    MacroDefinition,
    Record,
    Typedef,
)

try:
    # module to calculate word distances
    import Levenshtein

    HAVE_LEVENSHTEIN = True
except ImportError:
    HAVE_LEVENSHTEIN = False


# flake8: noqa: C901
# TODO break function apart to reduce complexity
def generate_cuda_interop_module_files(
    output_dir: str,
    cuda_global_module_name: str,
    hip_generator: CythonModuleGenerator,
    hip2cuda: dict,
    license_text: str,
    warn: bool = True,
    cuda_cmodule_prefix="cy",
    extra_cimports="",
    extra_imports="",
    extra_cmodule_cimports="",
):
    """Renders the Cython and Python module files that delegate CUDA Python
    API expressions to HIP Python.

    Args:
        output_dir (str):
            Root output directory. Subfolders 'hip-python-as-cuda/cuda` must exist in the that directory.
        cuda_global_module_name (str):
            The global module name of the CUDA module whose files are generated.
            Something like 'cuda.cudart' or 'cuda.bindings.driver'.
        generator (CythonModuleGenerator):
            A module that allows us to access the the parse tree of a HIP translation unit.
        hip2cuda (dict):
            A dictionary that maps HIP names to CUDA names.
        warn (bool, optional):
            _description_. Defaults to True.
        extra_cimports (str, optional):
            Additional Cython cimport statements.
            Use it if HIP and CUDA do declare certain types and functions in different header files.
            To give an example: CUjitInputType, which is linked to HIPRTC, is not part of NVRTC but of CUDART.
            Defaults to "".
        extra_imports (str, optional):
            Additional Python import statements.
            Use it if HIP and CUDA do declare certain types and functions in different header files.
            To give an example: CUjitInputType, which is linked to HIPRTC, is not part of NVRTC but of CUDART.
            Defaults to "".
        extra_cmodule_cimports (str, optional):
            Additional Cython cimport statements for the c-prefixed C module.
            Use it if HIP and CUDA do declare certain types and functions in different header files.
            To give an example: CUjitInputType, which is linked to HIPRTC, is not part of NVRTC but of CUDART.
            Defaults to "".
    """
    global HAVE_LEVENSHTEIN

    cuda_global_module_as_tuple = cuda_global_module_name.split(".")
    # Modern layout: <repo_root>/packages/hip-python-interop/src/<cuda parts>/
    cuda_parent_package_dir = os.path.join(
        output_dir, "packages", "hip-python-interop", "src",
        *(cuda_global_module_as_tuple[:-1]),
    )
    cuda_parent_package = ".".join(cuda_global_module_as_tuple[:-1])
    cuda_module_name = cuda_global_module_as_tuple[-1]

    indent = " " * 4

    hip_module_name = hip_generator.module_name
    # Use short aliases (e.g. `cyhip`, `hip`) so generated bodies can refer to
    # types as `cyhip.X` / `hip.X` instead of the long dotted form.
    hip_cmodule_alias = f"cy{hip_module_name}"
    hip_module_alias = hip_module_name
    hip_cmodule_name = f"rocm.bindings.{hip_cmodule_alias}"
    hip_module_cimport_name = f"rocm.bindings.{hip_module_alias}"
    hip_backend = hip_generator.backend

    c_interface_decl_part = [
        license_text,
        textwrap.dedent(
            f"""\

            cimport {hip_cmodule_name} as {hip_cmodule_alias}
            """
        )
        + extra_cmodule_cimports,
    ]
    python_interface_decl_part = [
        license_text,
        textwrap.dedent(
            f"""\

            __author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

            cimport {hip_cmodule_name} as {hip_cmodule_alias}
            cimport {hip_module_cimport_name} as {hip_module_alias}
            """
        )
        + extra_cimports,
        f"cimport {cuda_parent_package}.{cuda_cmodule_prefix}{cuda_module_name}",  # for checking compiler errors
    ]

    python_interface_impl_part_preamble = (
        license_text
        + textwrap.dedent(
            f"""\

            \"""
            Attributes:
            [ATTRIBUTES]
            \"""

            __author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

            import os
            import enum

            from rocm.bindings import {hip_module_name}  # makes {hip_module_name} types and routines accessible without import
                                                       # allows checks such as `hasattr(cuda.{cuda_module_name},"{hip_module_name}")`

            hip_python_mod = {hip_module_name}
            globals()["HIP_PYTHON"] = True
            """
        )
        + extra_imports
    )
    python_interface_impl_part = [
        textwrap.dedent(
            """\
            def _hip_python_get_bool_environ_var(env_var, default):
                yes_vals = ("true", "1", "t", "y", "yes")
                no_vals = ("false", "0", "f", "n", "no")
                value = os.environ.get(env_var, default).lower()
                if value in yes_vals:
                    return True
                elif value in no_vals:
                    return False
                else:
                    allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
                    raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")
            """
        ),
    ]

    # impl part is always empty
    def warn_(hip_name):
        global HAVE_LEVENSHTEIN
        msg = f"hipify-perl: no CUDA symbol found for HIP symbol {hip_name}"
        if HAVE_LEVENSHTEIN:
            cutoff = 0.9
            candidates = []
            for other_hip_name in hip2cuda:
                if (
                    Levenshtein.ratio(
                        hip_name,
                        other_hip_name,
                        processor=lambda tk: tk.lower(),  # do everything in lowercase
                        score_cutoff=cutoff,  # everything below cutoff is set to 0
                    )
                    > 0
                ):
                    candidates.append(other_hip_name)
            candidates_formatted = ", ".join(
                ["'" + c + "'" for c in candidates]
            )
            msg += f"; most similar hipify-perl HIP symbols (Levenshtein ratio > {cutoff}): [{candidates_formatted}]"
        _log.warning(msg)

    all = ["HIP_PYTHON", "hip_python_mod", hip_module_name]
    docstring_attributes = []
    # Sequence of (cuda_name, kind, payload) tuples used to render the
    # cuda binding's `.pyi` stub at the end of this function. `payload`
    # depends on `kind`:
    #   'class'    -> (hip_node, base_class_str)  - cdef class CUDA_X(hip.Y)
    #   'function' -> hip_node                    - cuda_name = hip.cuda_name
    #   'macro'    -> hip_node                    - cuda_name: Any
    #   'plain'    -> None                        - cuda_name: Any (no hip node)
    cuda_pyi_entries = []
    docstring_attributes.append(
        textwrap.dedent(
            f"""\
        HIP_PYTHON ({python_interface_pyobj_role_template.format(name="bool")}):
            `True`.
        hip_python_mod (module):
            A reference to the module {python_interface_pyobj_role_template.format(name=f"rocm.bindings.{hip_module_name}")}.
        {hip_module_name} (module):
            A reference to the module {python_interface_pyobj_role_template.format(name=f"rocm.bindings.{hip_module_name}")}.
        """
        )
    )

    def handle_enum_(node, hip_name, cuda_name, cuda_idx):
        nonlocal indent
        nonlocal all
        nonlocal docstring_attributes
        nonlocal c_interface_decl_part
        nonlocal python_interface_impl_part
        enum = node if isinstance(node, Enum) else node.lookup_innermost_type()
        c_constants = []
        python_constants = []
        for child_cursor in enum.cursor.get_children():
            hip_constant_name = child_cursor.spelling
            # append hip constant too, to help workarounds
            c_constants.append(
                f"from {hip_cmodule_name} cimport {hip_constant_name}"
            )
            python_constants.append(
                f"{hip_constant_name} = {hip_cmodule_alias}.{hip_constant_name}"
            )
            if hip_constant_name in hip2cuda:
                for cuda_constant_name in hip2cuda[hip_constant_name]:
                    c_constants.append(
                        f"from {hip_cmodule_name} cimport {hip_constant_name} as {cuda_constant_name}"
                    )
                    python_constants.append(
                        f"{cuda_constant_name} = {hip_cmodule_alias}.{hip_constant_name}"
                    )
            else:
                warn_(hip_constant_name)
        if isinstance(node, AnonymousEnum):  # cannot be typedefed
            python_interface_impl_part += python_constants
            all += [
                ln.split("=")[0].strip() for ln in python_constants
            ]  # recover cuda names
            for ln in python_constants:
                cuda_pyi_entries.append(
                    (ln.split("=")[0].strip(), "plain", None)
                )
        else:
            python_enum_metaclass_name = f"_{cuda_name}_EnumMeta"
            python_enum_hallucinate_var_name = (
                f"HIP_PYTHON_{cuda_name}_HALLUCINATE"
            )
            all.append(python_enum_metaclass_name)
            cuda_pyi_entries.append(
                (python_enum_hallucinate_var_name, "plain", None)
            )

            attribute = textwrap.dedent(
                f"""\
                 {python_enum_hallucinate_var_name}:
                     Make {python_interface_pyobj_role_template.format(name=cuda_name)} hallucinate values for non-existing enum constants. Disabled by default
                     if default is not modified via environment variable.

                     Default value can be set/unset via environment variable ``{python_enum_hallucinate_var_name}``.

                     * Environment variable values that result in `True` are: ``yes``, ``1``, ``y``, ``true``
                     * Those that result in `False` are: ``no``, ``0``, ``n``, ``false``.
                 """
            )
            all.append(python_enum_hallucinate_var_name)
            docstring_attributes.append(attribute)

            python_enum_metaclass = textwrap.dedent(
                f"""\
                {python_enum_hallucinate_var_name} = _hip_python_get_bool_environ_var("{python_enum_hallucinate_var_name}","false")

                class {python_enum_metaclass_name}(enum.EnumMeta):

                    def __getattribute__(cls,name):
                        global _get_hip_name
                        global {python_enum_hallucinate_var_name}
                        try:
                            result = super().__getattribute__(name)
                            return result
                        except AttributeError as ae:
                            if not {python_enum_hallucinate_var_name}:
                                raise ae
                            else:
                                used_vals = list(cls._value2member_map_.keys())
                                if not len(used_vals):
                                    raise ae
                                new_val = min(used_vals)
                                while new_val in used_vals: # find a free enum value
                                    new_val += 1

                                class HallucinatedEnumConstant():
                                    \"""Mimicks the orginal enum type this is derived from.
                                    \"""
                                    def __init__(self):
                                        pass

                                    @property
                                    def name(self):
                                        return self._name_

                                    @property
                                    def value(self):
                                        return self._value_

                                    def __eq__(self,other):
                                        if isinstance(other,{hip_module_name}.{hip_name}):
                                            return self.value == other.value
                                        return False

                                    def __repr__(self):
                                        \"""Mimicks enum.Enum.__repr__\"""
                                        return "<%s.%s: %r>" % (
                                                self.__class__._name_, self._name_, self._value_)

                                    def __str__(self):
                                        \"""Mimicks enum.Enum.__str__\"""
                                        return "%s.%s" % (self.__class__._name_, self._name_)

                                    def __hash__(self):
                                        return hash(str(self))

                                    @property
                                    def __class__(self):
                                        \"""Make this type appear as a constant of the actual
                                        CUDA enum type in isinstance checks.
                                        \"""
                                        return {cuda_name}
                                setattr(HallucinatedEnumConstant,"_name_",name)
                                setattr(HallucinatedEnumConstant,"_value_",new_val)
                                return HallucinatedEnumConstant()
                """
            )
            python_enum_class = textwrap.dedent(
                f"""
                class {cuda_name}({hip_module_name}.{enum.python_base_class_name},metaclass={python_enum_metaclass_name}):
                """
            )
            all.append(cuda_name)
            python_enum_class += textwrap.indent(
                "\n".join(python_constants), indent
            )

            python_interface_impl_part.append(python_enum_metaclass)
            python_interface_impl_part.append(python_enum_class)
            # The cuda enum class subclasses the hip enum's Python base.
            # Record both that and the metaclass for the .pyi.
            cuda_pyi_entries.append(
                (
                    cuda_name,
                    "class",
                    (
                        node,
                        f"{hip_module_alias}.{enum.python_base_class_name}",
                    ),
                )
            )
            cuda_pyi_entries.append(
                (python_enum_metaclass_name, "plain", None)
            )

        if isinstance(node, Enum) and cuda_idx == 0:
            if not isinstance(node, AnonymousEnum):
                c_interface_decl_part.append(
                    f"from {hip_cmodule_name} cimport {hip_name} as {cuda_name}"
                )
            c_interface_decl_part += c_constants
        else:  # if it is a typedef or there are multiple CUDA names
            hip_underlying_type_name = enum.name
            if hip_underlying_type_name in hip2cuda:
                cuda_underlying_type_name = hip2cuda[hip_underlying_type_name][
                    0
                ]  # take first
                cython_enum = (
                    f"ctypedef {cuda_underlying_type_name} {cuda_name}"
                )
                c_interface_decl_part.append(cython_enum)
            else:
                warn_(hip_underlying_type_name)

    # main loop over nodes
    for node in hip_backend.walk_filtered_nodes():
        hip_name = node.renamer(node.name)
        if isinstance(node, AnonymousEnum):
            # Anonymous enums won't have a different CUDA name but their constants might
            handle_enum_(
                node, hip_name, hip_name
            )  # hip_name is auto_generated in this case
        if hip_name in hip2cuda:
            cuda_names = hip2cuda[hip_name]
            for cuda_idx, cuda_name in enumerate(cuda_names):
                if isinstance(node, Enum) or (
                    isinstance(node, Typedef)
                    and node.is_pointer_to_enum(degree=(0, -1))
                ):
                    # enums require special care as they are modelled as "class <type>"
                    # and not as "cdef class" in the Python interface, just like in CUDA Python.
                    handle_enum_(node, hip_name, cuda_name, cuda_idx)
                elif (
                    isinstance(
                        node,
                        (
                            MacroDefinition,
                            Function,
                        ),
                    )
                    or isinstance(node, Typedef)
                    and (
                        node.is_pointer_to_record(degree=(0, -1))
                        or node.is_pointer_to_basic_type(degree=-1)
                        or node.is_pointer_to_void(degree=-1)
                    )
                ):
                    # These are Python objects/functions in the Python interface
                    c_interface_decl_part.append(
                        f"from {hip_cmodule_name} cimport {hip_name} as {cuda_name}"
                    )
                    docstring_attributes += [
                        (cuda_name, hip_module_name, hip_name),
                    ]
                    python_interface_impl_part += [
                        f"{cuda_name} = {hip_module_name}.{hip_name}"
                    ]
                    all.append(cuda_name)
                    if isinstance(node, Function):
                        cuda_pyi_entries.append(
                            (cuda_name, "function", node)
                        )
                    elif isinstance(node, MacroDefinition):
                        cuda_pyi_entries.append(
                            (cuda_name, "macro", node)
                        )
                    else:  # Typedef pointing to record/basic/void
                        cuda_pyi_entries.append(
                            (cuda_name, "plain", None)
                        )
                elif isinstance(node, Typedef) and (
                    node.is_pointer_to_basic_type(degree=(0, -1))
                    or node.is_pointer_to_void(degree=(0, -1))
                ):
                    canonical_type = node.cursor.type.get_canonical().spelling
                    c_interface_decl_part.append(
                        f"ctypedef {canonical_type} {cuda_name}"
                    )
                elif isinstance(node, (FunctionPointer, Record)):
                    # These are cdef classes ("extension types").
                    # So Python interface declaration must be cimported.
                    # and a subclass needs to be created to define a Python object. (TODO other options?)
                    c_interface_decl_part.append(
                        f"from {hip_cmodule_name} cimport {hip_name} as {cuda_name}"
                    )
                    #
                    cdef_subclass = f"cdef class {cuda_name}({hip_module_alias}.{hip_name}):\n{indent}pass"
                    python_interface_decl_part.append(cdef_subclass)
                    python_interface_impl_part.append(cdef_subclass)
                    all.append(cuda_name)
                    cuda_pyi_entries.append(
                        (
                            cuda_name,
                            "class",
                            (node, f"{hip_module_alias}.{hip_name}"),
                        )
                    )
        elif warn:
            warn_(hip_name)

    python_interface_decl_path = os.path.join(
        cuda_parent_package_dir, f"{cuda_module_name}.pxd"
    )
    python_interface_impl_path = os.path.join(
        cuda_parent_package_dir, f"{cuda_module_name}.pyx"
    )
    c_interface_decl_path = os.path.join(
        cuda_parent_package_dir, f"{cuda_cmodule_prefix}{cuda_module_name}.pxd"
    )
    with open(c_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(c_interface_decl_part))
    with open(python_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(python_interface_decl_part))
    with open(python_interface_impl_path, "w") as outfile:
        DOCSTRING_ATTRIBS = ""
        for attribute in docstring_attributes:
            if isinstance(attribute, tuple):
                cuda_name, hip_module_name, hip_name = attribute
                docstring_attrib = textwrap.dedent(
                    f"""\
                        {cuda_name}:
                            Alias of {python_interface_pyobj_role_template.format(name=hip_name)}
                        """
                )
            else:  # raw string
                docstring_attrib = attribute
            DOCSTRING_ATTRIBS += textwrap.indent(docstring_attrib, " " * 4)

        python_interface_impl_part.insert(
            0,
            python_interface_impl_part_preamble.replace(
                "[ATTRIBUTES]", DOCSTRING_ATTRIBS
            ),
        )
        outfile.write(
            "\n".join(python_interface_impl_part).rstrip()
            + "\n\n"
            + "__all__ = [\n"
            + "\n".join([f'    "{e}",' for e in all])
            + "\n]"
        )

    # ----- .pyi type-stub for the cuda binding -----
    # Each cuda symbol is an alias of a hip symbol (or constructed
    # locally — see 'plain' entries). Reuse the hip node's
    # `render_pyi_stub` so the cuda alias inherits the hip docstring
    # and signature verbatim. `base=` makes record/enum stubs render
    # as `class CUDA_X(hip.X):`, matching the .pyx where the cuda
    # class is `cdef class CUDA_X(hip.X): pass`.
    hip_cprefix = f"{hip_cmodule_alias}."
    pyi_lines = [
        "# AUTO-GENERATED by the hip-python code generator.",
        f"# Type stubs for {cuda_global_module_name}. Edits will be overwritten.",
        "",
        "from typing import Any",
        f"from rocm.bindings import {hip_module_name} as {hip_module_alias}",
        "",
    ]
    seen_names = set()
    pyi_all = []
    for entry_name, kind, payload in cuda_pyi_entries:
        if entry_name in seen_names or not entry_name.isidentifier():
            continue
        seen_names.add(entry_name)
        if kind == "plain" or payload is None:
            pyi_lines.append(f"{entry_name}: Any")
        elif kind == "class":
            hip_node, base = payload
            stub = hip_node.render_pyi_stub(
                hip_cprefix, override_name=entry_name, base=base,
            )
            if stub:
                pyi_lines.extend(stub)
            else:
                pyi_lines.append(
                    f"class {entry_name}({base}):"
                )
                pyi_lines.append(
                    "    def __init__(self, *args, **kwargs): ..."
                )
        elif kind == "function":
            stub = payload.render_pyi_stub(
                hip_cprefix, override_name=entry_name,
            )
            if stub:
                pyi_lines.extend(stub)
            else:
                pyi_lines.append(
                    f"def {entry_name}(*args, **kwargs): ..."
                )
        elif kind == "macro":
            stub = payload.render_pyi_stub(
                hip_cprefix, override_name=entry_name,
            )
            if stub:
                pyi_lines.extend(stub)
            else:
                pyi_lines.append(f"{entry_name}: Any")
        pyi_lines.append("")
        pyi_all.append(entry_name)
    if pyi_all:
        pyi_lines.append("__all__ = [")
        for n in sorted(pyi_all):
            pyi_lines.append(f"    {n!r},")
        pyi_lines.append("]")
    pyi_lines.append("")
    pyi_path = os.path.join(
        cuda_parent_package_dir, f"{cuda_module_name}.pyi"
    )
    with open(pyi_path, "w") as outfile:
        outfile.write("\n".join(pyi_lines))
