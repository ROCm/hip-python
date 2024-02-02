# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

"""Types for extracting information from HIP C++ sources.

This module defines types for extracting information such 
as device function definitions from a HIP C++ source file.
The `~.HIPSource` datatype allows to generate stubs and 
render wrapper functions for all device function declarations/definitions
in the file.
"""

import re
import textwrap
import logging

import rocm.clang.cindex as ci

from . import cparser

_log = logging.getLogger(__name__)


class HIPDeviceFunction:
    """HIP device function definition or declaration.

    This class provides access to properties of
    device functions that have been expressed
    as HIP C++ functions.

    Class attributes:
        TYPE_MAPPER:
            A callback for mapping LLVM clang binding types
            to other types. Default is the identity operation.
    """

    @staticmethod
    def DEFAULT_TYPE_MAPPER(clang_type: ci.Type):
        """Simply returns the clang_type"""
        return clang_type

    @staticmethod
    def TYPE_RENDERER(clang_type: ci.Type):
        """ """
        try:
            type = ".".join(
                tl.spelling
                for tl in cparser.TypeHandler(clang_type).clang_type_layer_kinds(
                    canonical=True
                )
            )
            if type.endswith("Record"):
                innermost = [
                    cparser.TypeHandler(clang_type).walk_clang_type_layers(
                        canonical=True
                    )
                ][-1]
                type = type.replace("Record", f"<{innermost.get_canonical().spelling}>")
            return type
        except Exception as e:
            return f"<{clang_type.get_canonical().spelling}>"

    TYPE_MAPPER = DEFAULT_TYPE_MAPPER

    @staticmethod
    def match(cursor: ci.Cursor):
        """Checks if the cursor is a function with device attribute."""
        if cursor.kind == ci.CursorKind.FUNCTION_DECL:
            for child in cursor.get_children():
                if child.kind == ci.CursorKind.CUDADEVICE_ATTR:
                    return True
        return False

    def __init__(self, cursor: ci.Cursor):
        self._cursor = cursor

    def parm_cursors(self):
        """Yields cursors of kind `~.ci.CursorKind.PARM_DECL`."""
        for child in self._cursor.get_children():
            if child.kind == ci.CursorKind.PARM_DECL:
                yield child

    def parm_types(self, canonical=False):
        """Yields the type of cursors of kind `~.ci.CursorKind.PARM_DECL`."""
        for cursor in self.parm_cursors():
            if canonical:
                yield cursor.type.get_canonical()
            else:
                yield cursor.type

    def parm_type_kinds(self, canonical=False):
        """Yields the type kind of cursors of kind `~.ci.CursorKind.PARM_DECL`."""
        for parm_type in self.parm_types(canonical=canonical):
            yield cparser.clang_type_kind(parm_type)

    def parm_type_kind_layers(self, canonical=False):
        """Yields the kind of the type layers of cursors of kind `~.ci.CursorKind.PARM_DECL`."""
        for parm_type in self.parm_types(canonical=canonical):
            yield list(cparser.TypeHandler(parm_type).walk_clang_type_layers())

    def result_type(self, canonical=False):
        """Returns the type of the result."""
        if canonical:
            return self._cursor.result_type.get_canonical()
        else:
            return self._cursor.result_type

    def result_type_kind(self, canonical=False):
        """Returns the kind of the result type."""
        return cparser.clang_type_kind(self.result_type(canonical=canonical))

    def result_type_kind_layers(self, canonical=False):
        """Returns the kind of each result type layer."""
        return list(
            cparser.TypeHandler(
                self.result_type(canonical=canonical)
            ).walk_clang_type_layers()
        )

    @property
    def is_definition(self):
        """If this is a function definition, i.e., there is a compound statement."""
        for child in self._cursor.get_children():
            if child.kind == ci.CursorKind.COMPOUND_STMT:
                return True
        return False

    @property
    def is_declaration(self):
        """If this is a function declaration, i.e., there is no compound statement."""
        return not self.is_definition

    @property
    def mangled_name(self):
        """The mangled name of the device function ("C++ name")."""
        return self._cursor.mangled_name

    @property
    def name(self):
        """Spelled name of the device function ("C name")."""
        return self._cursor.spelling

    @property
    def displayname(self):
        """Spelled function name plus argument list.

        In other words, returns the signature of the function
        sans return value and atributes.
        """
        return self._cursor.displayname

    @property
    def location(self):
        """Location as string expression '<filepath>:<line>:<column>'"""
        return f"{self._cursor.location.file}:{self._cursor.location.line}:{self._cursor.location.column}"

    @property
    def parm_names(self):
        """Names of all parameters as `list`."""
        return list(p.spelling for p in self.parm_cursors())

    @property
    def retval_type(self):
        """The return value type, up to `~.HipDeviceFunction.TYPE_MAPPER`.

        See:
            `~.TYPE_MAPPER`
        """
        self.TYPE_MAPPER(self._cursor.result_type)  # TODO

    @property
    def parm_type(self, name):
        """
        The type of a , up to `TYPE_MAPPER`.
        See:
            `~.HipDeviceFunction.parm_names`, `~.TYPE_MAPPER`.
        """
        for parm_decl in self.parm_cursors():
            if name == parm_decl.spelling:
                return self.TYPE_MAPPER(parm_decl)
        raise KeyError(f"parm '{name}' could not be found")

    def render_wrapper_function(self, internal_ns: str = "", prefix: str = ""):
        """Renders a wrapper function that calls another function with the same name and arguments.

        Renders a wrapper function that calls another function with the same name and arguments.
        The wrapper function is named as the mangled name of the other function plus a user-defined prefix.
        If the called other function is placed into a namespace then it can be provided
        to this render call via the ``internal_ns`` argument.

        Background:
            Often HIP C++ device code libraries set inline and force-inline attributes to device functions.
            If not called by any other non-inline device function or kernel inside the HIP C++ file,
            these definitions will thus not appear in an LLVM module generated from that file.
            Rendering a wrapper non-inline wrapper function allows us to "get" the code of
            those functions into an LLVM module.

        Note:
            At least one of the arguments ``prefix`` and ``internal_ns`` must not be `None`
            or an empty string. Otherwise, an exception is raised.

        Example:

            Given a function with signature ``void foo(int a,int b)`` (excluding attributes), ``prefix="TEST_"`` and ``internal_ns="test"``,
            this function returns the following string:

            ```c++
            extern "C" __attribute__((device)) void TEST__Z3fooii(int a, int b) {
                test::foo(a,b);
            }
            ```

        Args:
            prefix (`str`):
                A prefix to give the wrapper functions to prevent conflicts
                with the wrapped functions. Defaults to ``""``.
                Note that at least one of the arguments ``prefix`` and ``internal_ns``
                must not be `None` or an empty string.
            internal_ns (`str`):
                If the called other function is placed into a namespace then the namespace expression
                can be provided via this argument. Defaults to ``""``.
                Note that at least one of the arguments ``prefix`` and ``internal_ns``
                must not be `None` or an empty string.
                to this render call via the ``internal_ns`` argument.
        """
        if not prefix and not internal_ns:
            raise ValueError(
                "one of arguments 'prefix' and 'internal_ns' must not be 'None' or empty string"
            )
        if not internal_ns.endswith("::"):
            internal_ns = internal_ns + "::"
        rettype = self._cursor.result_type.spelling
        argnames = ",".join([f"_{i}" for i, p in enumerate(self.parm_cursors())])
        arglist = ",".join(
            [f"{p.type.spelling} _{i}" for i, p in enumerate(self.parm_cursors())]
        )
        return textwrap.dedent(
            f"""\
            extern "C" __attribute__((device)) {rettype} {prefix}{self.mangled_name}({arglist}) {{
                return {internal_ns}{self.name}({argnames});
            }}
            """
        )


class HIPSource:
    """Parse HIP C++ source and generate stubs."""

    def __init__(
        self,
        source: str,
        filename: str = "source.hip",
        append_cflags: list = [],
        filter=lambda cursor: True,
    ):
        """Constructor.

        Args:
            source (str): Contents of a HIP C++ source file.
            filename (str): Name for the HIP C++ source file, defaults to "source.hip".
            append_cflags (list, optional):
                Additional compiler flags to append, e.g., to influence the preprocessing of
                the parsed file.
            filter (optional): Filter for removing unwanted entities. Defaults to ``lambda cursor:True``.
        """
        self.source = source
        self.filename = filename
        self._hip_device_functions = self._parse_hip_source(
            source, filename, append_cflags, filter
        )

    @property
    def device_functions(self):
        return self._hip_device_functions

    def check_for_duplicates(self, log_errors=False, remove=False):
        """Checks all HIP entities for duplicates.

        Args:
            remove (bool, optional):
                Remove the duplicates.In this case,
                no errors are logged and no exception will be raised.
                Defaults to False.
            log_errors (bool, optional):
                Log an error message per duplicate. Defaults to False.
        Raises:
            `RuntimeError`: If duplicates are detected and argument `remove` is set to False.
        """
        found_duplicate = False
        for _, variants in self.device_functions.items():
            variants_copy = list(variants)  # shallow copy
            for i, device_fun1 in enumerate(variants_copy):
                for device_fun2 in variants_copy[i + 1 :]:
                    if (
                        device_fun1 != device_fun2
                        and device_fun1.mangled_name == device_fun2.mangled_name
                        and device_fun1.is_definition == device_fun2.is_definition
                    ):
                        found_duplicate = True
                        if log_errors:
                            _log.error(
                                f"{device_fun1.mangled_name} at {device_fun1.location} (): duplicate at {device_fun2.location}"
                            )
                        if remove:
                            variants.remove(device_fun2)
        if not remove and found_duplicate:
            raise RuntimeError("found duplicates")

    def _parse_hip_source(
        self,
        source: str,
        filename: str = "source.hip",
        append_cflags: list = [],
        filter=lambda cursor: True,
    ):
        """Parse the HIP C++ source.

        Parse the HIP C++ source and collect entities that
        we are interested in such as device functions.
        """
        hip_device_functions = {}

        parser = cparser.CParser(
            filename="source.hip",
            append_cflags=["-x", "hip"] + append_cflags,
            unsaved_files=[(filename, source)],
        )
        parser.parse()
        for cursor in parser.cursor.get_children():
            filename: str = cursor.location.file
            if filter(cursor):
                if HIPDeviceFunction.match(cursor):
                    if not cursor.spelling in hip_device_functions:
                        hip_device_functions[cursor.spelling] = []
                    hip_device_functions[cursor.spelling].append(
                        HIPDeviceFunction(cursor)
                    )
        return hip_device_functions

    def create_stubs(
        self,
        stub_base_class=object,
        function_renamer_splitter=lambda name: [name],
        stub_processor=lambda stub, parent, device_fun_variants, name_parts: None,
    ) -> dict:
        """Generate stubs from the device functions and postprocess them.

        function_renamer_splitter (optional):
            A callback for renaming and splitting function names.
            If the function name is splitted, stups will
            Defaults to ``lambda name:[name]``.
        stub_processor (optional):
            Specify a callback that allows to postprocess the stub and its top-most
            parent. The callback gets the arguments stub type, parent type, device function variants
            (`list` of `~.HIPDeviceFunction`) and the name parts (`list` of `str`).
            The stub is associated with last entry in ``name_parts``, the parent with the first.
            Defaults to no action.

        See:
            `~.HIPDeviceFunction`.

        Returns:
            `dict`: A `dict` that maps function name to one or more ``HipDeviceFunction`` objects.`
        """

        def descend_(parts, thedict, variants, i=0, parent=None):
            nonlocal stub_processor

            cls = thedict.get(parts[i], type(parts[i].lower(), (stub_base_class,), {}))
            if i < len(parts) - 1:
                member = descend_(
                    parts, cls.__dict__, variants, i + 1, parent if parent else cls
                )  # sets parent to top-most parent for i > 0
                setattr(cls, parts[i + 1], member)
            else:  # i == len(parts)
                _log.debug(f"created stub '{'.'.join(parts)}'")  # TODO warn -> debug
                stub_processor(cls, parent if parent else cls, variants, parts)
            return cls

        result = {}
        for name, variants in self.device_functions.items():
            parts = function_renamer_splitter(name)
            result[parts[0]] = descend_(parts, result, variants)
        return result

    def render_device_function_wrappers(
        self,
        internal_ns: str = "",
        prefix: str = "",
        wrapper_for_declarations: bool = False,
    ):
        """Per device function, renders a wrapper function
        Args:
            prefix (`str`):
                A prefix to give the wrapper functions to prevent conflicts
                with the wrapped functions. Defaults to ``""``.
                Note that at least one of the arguments ``prefix`` and ``internal_ns``
                must not be `None` or an empty string.
            internal_ns (`str`):
                If the called other function is placed into a namespace then the namespace expression
                can be provided via this argument. Defaults to ``""``.
                Note that at least one of the arguments ``prefix`` and ``internal_ns``
                must not be `None` or an empty string.
                to this render call via the ``internal_ns`` argument.
            wrapper_for_declarations (`bool`, optional):
                If a wrapper should also be generated for declarations.
                Defaults to `False`.
        Note:
            If a declaration and definition is present for a
            device function, only one wrapper is generated.

        Raises:
            `RuntimeError`:
                If more than one declaration has been found for a device function.
                If more than one definition has been for a device function.
        """
        if not prefix and not internal_ns:
            raise ValueError(
                "one of arguments 'prefix' and 'internal_ns' must not be 'None' or empty string"
            )
        result = ""
        for _, variants in self.device_functions.items():
            for mangled_name in set([v.mangled_name for v in variants]):
                existing_declaration = (
                    None  # we may have 1 declaration per mangled name
                )
                existing_definition = None  # we may have 1 definition per mangled name
                for device_fun in [
                    v for v in variants if v.mangled_name == mangled_name
                ]:
                    generate_wrapper = False
                    neither_found_yet = (existing_declaration == None) and (
                        existing_definition == None
                    )
                    if neither_found_yet and device_fun.is_declaration:
                        existing_declaration = device_fun
                        generate_wrapper = wrapper_for_declarations
                    elif neither_found_yet and device_fun.is_definition:
                        existing_definition = device_fun
                        generate_wrapper = True
                    elif existing_declaration and device_fun.is_declaration:
                        raise RuntimeError(
                            f"{existing_declaration.mangled_name} at {existing_declaration.location}: redeclared at '{device_fun.location}'"
                        )
                    elif existing_declaration and device_fun.is_definition:
                        generate_wrapper = not wrapper_for_declarations
                    elif existing_definition and device_fun.is_definition:
                        raise RuntimeError(
                            f"{existing_definition.mangled_name} at {existing_definition.location}: redefined at '{device_fun.location}'"
                        )
                    if generate_wrapper:
                        assert isinstance(device_fun, HIPDeviceFunction)
                        result += device_fun.render_wrapper_function(
                            prefix=prefix, internal_ns=internal_ns
                        )
        return result


# if __name__ == "__main__":
#     # TODO convert to test
#     import pprint

#     ci.Config.set_library_path("/opt/rocm/llvm/lib")
#     from rocm.amd_comgr import amd_comgr as comgr

#     import llvmutils
#     import comgrutils

#     HIPDeviceFunction.TYPE_MAPPER = HIPDeviceFunction.TYPE_RENDERER

#     def _hiprtc_runtime_header_filter(cursor: ci.Cursor):
#         return not cursor.spelling.startswith("operator")

#     def _hiprtc_runtime_function_renamer_splitter(name: str):
#         """Splits atomics, strips "_".

#         Examples:

#         * Converts ``"safeAtomicAdd"`` to ``["atomic","add","safe"]``.
#         * Converts ``"unsafeAtomicAdd_system"`` to ``["atomic","add","system","unsafe"]``.
#         * Converts ``"__syncthreads"`` to ``["syncthreads"]``.

#         Returns:
#             list: Parts of the name, describe a nesting hierarchy.
#         """
#         p_atomic = re.compile(r"(safe|unsafe)?[Aa]tomic([A-Z][A-Za-z]+)_?(\w+)?")
#         name = name.lstrip("_")
#         if "atomic" in name.lower():
#             name = p_atomic.sub(repl=r"atomic.\2.\3.\1", string=name).rstrip(".")
#             name = name.lower()
#         return name.split(".")

#     _hiprtc_runtime_hip_source = HIPSource(
#         source=comgr.ext.HIPRTC_RUNTIME_HEADER,
#         filter=_hiprtc_runtime_header_filter,
#         append_cflags=["-D__HIPCC_RTC__"],
#     )

#     # pprint.pprint(
#     #     hiprtlib.create_stubs(
#     #         function_renamer_splitter=_hiprtc_runtime_function_renamer_splitter
#     #     )
#     # )

#     PREFIX = "NUMBA_HIP_"
#     _hiprtc_runtime_hip_source.check_for_duplicates(log_errors=True)
#     wrappers = _hiprtc_runtime_hip_source.render_device_function_wrappers(prefix=PREFIX)

#     from hip import HIP_VERSION_TUPLE

#     coordinates = ""
#     for kind in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
#         for dim in "xyz":
#             coordinates += textwrap.dedent(
#                 f"""\
#             extern "C" std::uint32_t __attribute__((device)) {PREFIX}{kind}_{dim}() {{
#                 return {kind}.{dim};
#             }}
#             """
#             )

#     hipdevicelib_source = comgr.ext.HIPRTC_RUNTIME_HEADER + wrappers + coordinates

#     # print(hipdevicelib_source)
#     (bcbuf, logbuf, diagnosticbuf) = comgrutils.compile_hip_source_to_llvm(
#         amdgpu_arch="gfx90a",
#         extra_opts=" -D__HIPCC_RTC__",
#         hip_version_tuple=HIP_VERSION_TUPLE,
#         comgr_logging=False,
#         source=hipdevicelib_source,
#         to_llvm_ir=True,
#     )

#     with open("hiprtc_runtime2.ll", "w") as outfile:
#         outfile.write(bcbuf.decode("utf-8"))

#     with open("hiprtc_runtime2.ll", "r") as infile:
#         irbuf = infile.read()
#         llvmutils.convert_llvm_ir_to_bc(irbuf)
