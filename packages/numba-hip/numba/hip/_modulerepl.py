# MIT License
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sys
import types
import ast

AST_VERBOSE = False  # Verbose output when doing AST comparison


def create_module_from_snippet(
    module_content: str,
    context: dict = {},  # in
    preprocess: callable = lambda content: content,
):
    """Executes the module code in the given module context and then returns the module's dict.

    Args:
        preprocess (**callable**):
            Takes the file content of the original file and returns a modified file or
            the top AST node of the modified file.
    """
    module_dict = dict(context)
    content_preprocessed = preprocess(module_content)
    # if isinstance(content_derived,str): print(content_derived)
    exec(
        compile(content_preprocessed, f"<string> <modified>", "exec"), module_dict
    )  # populates module_context
    return module_dict


def load_module(
    module_path: str,
    context: dict = dict(),  # in
    preprocess: callable = lambda content: content,
):
    """Executes the module code in the given module context and then returns the module's dict.

    Args:
        preprocess (**callable**):
            Takes the file content of the original file and returns a modified file or
            the top AST node of the modified file.
    """
    with open(module_path, "r") as infile:  # must be read and not openend
        return create_module_from_snippet(
            infile.read(),
            context,
            preprocess,
        )


def create_derived_module(
    new_global_name: str,
    module_path_or_content: str,
    is_file_path: bool = True,
    context: dict = {},  # in
    preprocess: callable = lambda content: content,
):
    """Wraps result of `load_module`/`create_module_from_snippet` into a `types.ModuleType` with the given name."""
    new_module = types.ModuleType(new_global_name)
    if is_file_path:
        source_module_dict = load_module(module_path_or_content, context, preprocess)
    else:
        source_module_dict = create_module_from_snippet(
            module_path_or_content, context, preprocess
        )
    new_module.__dict__.update(source_module_dict)
    return new_module


def create_and_register_derived_module(
    new_global_name: str,
    module_path_or_content: str,
    is_file_path: bool = True,
    context: dict = dict(),  # in
    preprocess: callable = lambda content: content,
):
    """Directly registers result of `create_derived_module` in `sys.modules`."""
    if new_global_name in sys.modules:
        raise RuntimeError(
            f"there is already a module with name '{new_global_name}' in 'sys.modules'."
        )
    new_module = create_derived_module(
        new_global_name, module_path_or_content, is_file_path, context, preprocess
    )
    sys.modules[new_global_name] = new_module
    return new_module


def get_loc(node: ast.AST):
    """Puts the 'lineno' and 'col_offset' attributes of 'node' into a dict."""
    return dict(lineno=node.lineno, col_offset=node.col_offset)


def to_ast_node(expr: str, **kwattribs):
    """Renders a Python string into an AST."""
    expr: ast.AST = ast.parse(expr).body[0]
    if isinstance(expr, ast.Expr):
        expr: ast.AST = expr.value
    expr.__dict__.update(**kwattribs)
    return expr


def compare_ast_nodes(node: ast.AST, other: ast.AST):
    """Checks if the fields of the two nodes match, recursively.
    Does not compare any metadata such as location information.

    See:
        You can enable verbose output via `AST_VERBOSE`.
    """

    def compare_lists_(first, second):
        for i, a in enumerate(first):
            try:
                b = second[i]
            except IndexError:
                return False  # lists have not the same size
            mask = 2 * int(isinstance(a, ast.AST)) + int(isinstance(b, ast.AST))
            if mask == 0b11:
                if not compare_ast_nodes(a, b):
                    return False
            elif mask == 0b00:
                list_mask = 2 * int(isinstance(a, (tuple, list))) + int(
                    isinstance(b, (tuple, list))
                )
                if list_mask == 0b11:
                    if not compare_lists_(a, b):
                        return False
                elif list_mask == 0b00:
                    if a != b:
                        return False
                else:
                    return False
            else:
                return False
        return True

    gen_other = ast.iter_fields(other)
    for n_k, n_v in ast.iter_fields(node):
        try:
            o_k, o_v = next(gen_other)
        except StopIteration:
            return False  # other tree is smaller
        if n_k != o_k:
            return False  # keys do not match
        mask = 2 * int(isinstance(n_v, ast.AST)) + int(isinstance(o_v, ast.AST))
        if mask == 0b11:
            if AST_VERBOSE:
                print(f"{n_k}:{type(n_v)} vs {o_k}:{type(o_v)}")
            if not compare_ast_nodes(n_k, o_k):
                return False
        elif mask == 0b00:
            list_mask = 2 * int(isinstance(n_v, (tuple, list))) + int(
                isinstance(o_v, (tuple, list))
            )
            if list_mask == 0b11:
                if AST_VERBOSE:
                    print(f"{n_k}:{type(n_v)} vs {o_k}:{type(o_v)}'")
                if not compare_lists_(n_v, o_v):
                    return False
            elif list_mask == 0b00:
                if AST_VERBOSE:
                    print(f"{n_k}:'{str(n_v)}' vs {o_k}:'{str(o_v)}'")
                if n_v != o_v:
                    return False
            else:
                return False
        else:
            return False
    return True


class ModuleReplicator:
    """Helper class for replicating and modifiying modules/whole packages.

    Instances of this class need to be pointed to the package that contains the source files
    that they should replicate. This can be done via absolute paths or relative to the instantiator's location
    in the installation directory as shown below:

    ```py
    os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv")
    ```

    The class further needs to know the global identifier for the package, e.g.
    an expression like `numba.hip.cudadrv` in order to register newly created
    modules correctly in the `sys.modules` registry.

    Members:
        new_global_pkg_name (str):
                The global name of this package, e.g. "numba.hip.cudadrv".
        orig_pkg_path (str):
            Path to the sources files of the original package, e.g.
            `os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv")`
        base_context (dict):
            Base context that is passed to the exec call that loads the original module.
        preprocess_all (callable):
            Preprocessor that will be applied to all modules. Is applied before per-module preprocessing.
            Users must take care by themselves that input and output formats match (string vs AST).
            Defaults to the identity operation. If None is supplied, the identity operation
            is used too. Can be disabled via the `enable_preprocess_all` member.
        enable_preprocess_all (bool):
            If the `preprocess_all` routine should be applied or not. Defaults to True.
            Use it to turn the generic preprocessing on/off locally.
    """

    def __init__(
        self,
        new_global_pkg_name: str,
        orig_pkg_path: str,
        base_context: dict,
        preprocess_all: callable = lambda content: content,
    ):
        """Constructor.

        Args:
            new_global_pkg_name (str):
                The global name of this package, e.g. "numba.hip.cudadrv".
            orig_pkg_path (str):
                Path to the sources files of the original package, e.g.
                `os.path.join(os.path.dirname(__file__), "..", "..", "cuda", "cudadrv")`
            base_context (dict):
                This is where you pass the `globals()` of the caller.
            preprocess_all (callable):
                Preprocessor that will be applied to all modules. Is applied before per-module preprocessing.
                Users must take care by themselves that input and output formats match (string vs AST).
                Defaults to the identity operation.
        """
        self.new_global_pkg_name = new_global_pkg_name
        self.pkg_name = new_global_pkg_name.split(".")[-1]
        self.orig_pkg_path = orig_pkg_path
        self.base_context = dict(base_context)  # shallow copy
        self.preprocess_all = preprocess_all

        self.enable_preprocess_all = True

        self._clean_base_context()

        self.get_loc = get_loc
        self.to_ast_node = to_ast_node
        self.compare_ast_nodes = compare_ast_nodes

    def _clean_base_context(self):
        for k in ("__doc__", "__file__", "__path__", "__cached__"):
            if k in self.base_context:
                del self.base_context[k]

    def _finalize_module_context(self, module_context: dict, new_name: str):
        """Create the default context for the derived module."""
        module_context.update(
            __name__=f"{self.new_global_pkg_name}.{new_name}",  # if __main__
            __package__=self.pkg_name,
        )

    def _create_path(self, orig_name: str, ext=".py"):
        """Returns the path to the original module in the original package folder."""
        return os.path.join(self.orig_pkg_path, orig_name + ext)

    def create_derived_module(
        self,
        new_name: str,
        orig_name: str = None,
        from_file: bool = True,
        module_content: str = None,
        preprocess: callable = lambda content: content,
        extra_context: dict = {},
    ):
        """Wraps result of `load_module` into a `types.ModuleType` with the given `new_name`.

        Note:
            `orig_name` must only be specified if it differs from `new_name`.

        Args:
            new_name (str):
                The (local) name of the newly created module.
            orig_name (str, optional):
                The (local) name of the original module that we want to replicate. If `None` is passed, `new_name` is used. Defaults to None.
            preprocess (callable, optional):
                Preprocessing to apply to the content of the read file. Can return a `str` or an AST.
                Defaults to the identity operation.
            extra_context (dict, optional): Additional context. Defaults to `{}`.

        Returns:
            types.ModuleType: The new module object.
        """
        if orig_name == None:
            orig_name = new_name
        final_context = dict(self.base_context)
        final_context.update(extra_context)
        preprocess_all = (
            self.preprocess_all
            if (self.preprocess_all != None and self.enable_preprocess_all)
            else (lambda content: content)
        )
        if not from_file and not isinstance(module_content, str):
            raise ValueError(
                "'from_file' set to `False` but value of kwarg `module_content` is no `str`."
            )
        result = create_derived_module(
            f"{self.new_global_pkg_name}.{new_name}",
            self._create_path(orig_name) if from_file else module_content,
            is_file_path=from_file,
            context=final_context,
            preprocess=lambda content: preprocess(preprocess_all(content)),
        )
        self._finalize_module_context(result.__dict__, new_name)
        return result

    def create_and_register_derived_module(
        self,
        new_name: str,
        orig_name: str = None,
        from_file: bool = True,
        module_content: str = None,
        preprocess: callable = lambda content: content,
        extra_context: dict = {},
    ):
        """Directly registers result of `self.create_derived_module(...)` in `sys.modules`.

        Note:
            `orig_name` must only be specified if it differs from `new_name`.

        Args:
            new_name (str):
                The (local) name of the newly created module.
            orig_name (str, optional):
                The (local) name of the original module that we want to replicate. If `None` is passed,
                `new_name` is used. Defaults to None.
            from_file (bool, optional):
                If we load this module from a file in the original folder. Otherwise,
                `module_content` must be used to provide the file content directly.
                This can be used to create submodules with custom code in the derived package.
                Defaults to True.
            module_content (str, optional):
                In case `from_file` is set to `False`, this kwarg must be used to supply
                the content of the module file. This can be used to create submodules with
                custom code in the derived package. Defaults to None.
            preprocess (callable, optional):
                Preprocessing to apply to the content of the read file. Can return a `str` or an AST.
                Defaults to the identity operation.
            extra_context (dict, optional): Additional context. Defaults to `{}`.

        Returns:
            types.ModuleType: The new module object.
        """
        if orig_name == None:
            orig_name = new_name
        final_context = dict(self.base_context)
        final_context.update(extra_context)
        preprocess_all = (
            self.preprocess_all
            if (self.preprocess_all != None and self.enable_preprocess_all)
            else (lambda content: content)
        )
        if not from_file and not isinstance(module_content, str):
            raise ValueError(
                "'from_file' set to `False` but value of kwarg `module_content` is no `str`."
            )
        result = create_and_register_derived_module(
            f"{self.new_global_pkg_name}.{new_name}",
            self._create_path(orig_name) if from_file else module_content,
            is_file_path=from_file,
            context=final_context,
            preprocess=lambda content: preprocess(preprocess_all(content)),
        )
        self._finalize_module_context(result.__dict__, new_name)
        return result
