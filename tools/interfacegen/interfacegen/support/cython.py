#!/usr/bin/env python3
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

import argparse
from datetime import datetime
import os
import re
import textwrap
from pathlib import Path

from . import gitversion
from . import includetree as it

MIT_LICENSE_AMD = f"""\
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
"""


def get_bool_environ_var(env_var, default):
    yes_vals = ("true", "1", "t", "y", "yes")
    no_vals = ("false", "0", "f", "n", "no")
    value = os.environ.get(env_var, default).lower()
    if value in yes_vals:
        return True
    elif value in no_vals:
        return False
    else:
        allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals) + list(no_vals))])
        raise RuntimeError(
            f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}"
        )


def create_rocm_package_cli_parser(
    env_var_prefix: str, libs_example: str, *args, **kwargs
):
    parser = argparse.ArgumentParser(*args, **kwargs)

    def dir_path(arg):
        if not os.path.isdir(arg):
            raise NotADirectoryError(arg)
        return arg

    parser.add_argument(
        "output_dir",
        type=dir_path,
        help="The output directory to which the files should be written to. Must contain `rocm-llvm-python` subfolder.",
    )
    parser.add_argument(
        "--rocm-path",
        type=str,
        required=False,
        dest="rocm_path",
        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.",
    )

    def rocm_version(arg):
        if not re.match(r"[0-9]+\.[0-9]+\.[0-9]+", arg):
            raise ValueError(
                "Value of required argument `--rocm-version` must be a dot-separated number triple."
            )
        return [int(p) for p in arg.split(".")]

    parser.add_argument(
        "--rocm-version",
        type=rocm_version,
        required=True,
        dest="rocm_version",
        help="The ROCm version.",
    )
    parser.add_argument(
        "--clang-resource-dir",
        required=False,
        dest="clang_resource_dir",
        help=f"The clang resource directory. Can also be set via environment variable 'CLANG_RES_DIR'.",
    )
    parser.add_argument(
        "--libs",
        type=str,
        required=False,
        dest="libs",
        help=f"The libraries to generate interfaces for, as comma-separated list, e.g. '{libs_example}'. Pass '*' to generate all, pass '' to generate none. Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.",
    )
    parser.add_argument(
        "--no-rt-linking",
        required=False,
        action="store_false",
        dest="runtime_linking",
        help="If HIP libraries should not be linked at runtime by the HIP Python modules.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        action="store_true",
        dest="verbose",
        default=False,
        help="Verbose output.",
    )
    parser.set_defaults(
        rocm_path=os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME", None)),
        clang_resource_dir=os.environ.get(f"CLANG_RES_DIR", None),
        libs=os.environ.get(f"{env_var_prefix}_LIBS", "*"),
        runtime_linking=get_bool_environ_var(
            f"{env_var_prefix}_RUNTIME_LINKING", "true"
        ),
        verbose=False,
    )
    return parser


class PackageOpts:
    def __init__(self):
        self.package_dir: str = None
        self.rel_inc_dir: str = None
        self.abs_inc_dir: str = None
        self.env_var_prefix: str = None
        self.clang_res_dir: str = None
        self.runtime_linking: str = None
        self.generator_args: list = None
        self.libs_user_spec: str = None
        self.dll: str = None
        self.util_pkg: str = None
        self.author: str = None
        self.email: str = None


class RocmPackageOpts(PackageOpts):
    def __init__(self):
        PackageOpts.__init__(self)
        self.rocm_path: str = None
        self.rocm_version: tuple = (0, 0, 0)

    @property
    def author_email(self):
        return f"{self.author} <{self.email}>"

    @property
    def util_types_prefix(self):
        return self.util_pkg + ".types."

    @property
    def rocm_version_major(self):
        return self.rocm_version[0]

    @property
    def rocm_version_minor(self):
        return self.rocm_version[1]

    @property
    def rocm_version_patch(self):
        return self.rocm_version[2]


def create_rocm_package_opts_from_cli(
    project: str,
    env_var_prefix: str,
    libs_example: str,
    package: str,
    rel_inc_dir: str,
    dll: str,
    util_pkg: str,
    author: str,
    email: str,
) -> RocmPackageOpts:
    rocm_package_opts = RocmPackageOpts()
    rocm_package_opts.dll = dll
    rocm_package_opts.util_pkg = util_pkg
    rocm_package_opts.author = author
    rocm_package_opts.email = email
    rocm_package_opts.env_var_prefix = env_var_prefix

    parser = create_rocm_package_cli_parser(
        env_var_prefix,
        libs_example,
        description=textwrap.dedent(
            f"""\
        Generator for {project} package '{package}'.
    
        NOTE:
            You can also use the environment variables 'ROCM_PATH' (or 'ROCM_HOME'),
            'CLANG_RES_DIR', '{env_var_prefix}_LIBS',
            '{env_var_prefix}_RUNTIME_LINKING' instead of the command line interface.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parser.parse_args()

    rocm_package_opts.package_dir = os.path.join(args.output_dir, package)
    rocm_package_opts.runtime_linking = args.runtime_linking
    rocm_package_opts.libs_user_spec = args.libs
    rocm_package_opts.rocm_version = args.rocm_version

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    rocm_package_opts.rocm_path = args.rocm_path
    rocm_package_opts.abs_inc_dir = os.path.join(args.rocm_path, rel_inc_dir)

    if not args.clang_resource_dir:
        raise RuntimeError(
            textwrap.dedent(
                """\
            Clang resource directory is not set.
            
            Hint: If `clang` is in the PATH, you can 
            run `clang -print-resource-dir` to obtain the path to
            the resource directory.

            Hint: If you have the HIP SDK installed, you have `amdclang` installed in
            `ROCM_PATH/bin/`. You can use it to run the above command too.
            
            Hint: If you have the HIP SDK installed, the last include folder listed in ``hipconfig --cpp_config``
            points to the `amdclang` compiler's resource dir too.
            """
            )
        )
    rocm_package_opts.clang_res_dir = args.clang_resource_dir
    rocm_package_opts.generator_args = [
        f"-I{rocm_package_opts.abs_inc_dir}",
        "-resource-dir",
        args.clang_resource_dir,
    ]
    print(rocm_package_opts.generator_args)
    return rocm_package_opts


def user_specified_lib_names(avail_lib_names: list, user_spec: str):
    """

    Args:
        user_spec:
            The libraries to generate interfaces for, as comma-separated list, e.g. 'core,types'.
            Users pass '*' to generate all, pass '' to generate none.
            Users add a prefix '^' to NOT generate code for the comma-separated list of
            libraries that follows but all other libraries.
    """
    processed_libs = user_spec.replace(" ", "")
    if processed_libs == "*":
        lib_names = avail_lib_names
    else:
        if processed_libs.startswith("^"):
            processed_libs = processed_libs[1:].split(",")
            lib_names = [name for name in avail_lib_names if name not in processed_libs]
        else:
            processed_libs = processed_libs.split(",")
            lib_names = processed_libs
        for name in processed_libs:
            if name not in avail_lib_names:
                raise ValueError(
                    f"library name '{name}' is not valid, use one of: {', '.join(avail_lib_names)}"
                )
    return lib_names


def versions(rocm_version_major: int, rocm_version_minor: int, rocm_version_patch: int):
    rocm_version_name: str = (
        f"{rocm_version_major}.{rocm_version_minor}.{rocm_version_patch}"
    )
    rocm_version: int = (
        rocm_version_major * 10000000 + rocm_version_minor * 100000 + rocm_version_patch
    )

    version: str = f"{rocm_version_name}.{gitversion.git_branch_rev_count(gitversion.git_current_branch())}"
    long_version: str = (
        f"{rocm_version_name}.{gitversion.version(append_hash=True,append_date=True)}"
    )
    return (rocm_version, rocm_version_name, version, long_version)


def lstrip_all_lines(text: str, lstrip_chars: str = " "):
    return "".join(
        [line.lstrip(lstrip_chars) for line in text.splitlines(keepends=True)]
    )


def write_rocm_package_main_init_file(
    main_dir: str,
    author: str,  # e.g. "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
    rocm_version: tuple,
    local_module_names: list,
    epilog: str = "",
):
    (rocm_version_no, rocm_version_name, _, _) = versions(*rocm_version)
    with open(os.path.join(main_dir, "__init__.py"), "w") as f:
        f.write(
            lstrip_all_lines(
                f"""\
                {MIT_LICENSE_AMD}
            
                # This file has been autogenerated, do not modify.
                
                __author__ = "{author}"

                from ._version import *
                ROCM_VERSION = {rocm_version_no}
                ROCM_VERSION_NAME = rocm_llvm_version_name = "{rocm_version_name}"
                ROCM_VERSION_TUPLE = rocm_llvm_version_tuple = ({rocm_version[0]},{rocm_version[1]},{rocm_version[2]})

                """
            )
        )
        for mod in local_module_names:
            f.write(
                textwrap.dedent(
                    f"""\
                try:
                    from . import {mod}
                except ImportError:
                    pass # may have been excluded from build
                """
                )
            )
        f.write(epilog)


def write_version_py_in(
    main_dir: str,
    global_var_prefix: str,  # e.g. ROCM_LLVM_PYTHON
    author: str,
    license_text: str,
    version: str,
    long_version: str,
    global_module_names: list,
):
    module_list_as_str = "\n".join(f'\t"{n}",' for n in global_module_names)
    with open(os.path.join(main_dir, "_version.py.in"), "w") as f:
        f.write(
            lstrip_all_lines(
                f"""\
            {license_text}
            
            # This file has been autogenerated, do not modify.
            
            __author__ = "{author}"

            VERSION = __version__ = "{version}.{{VERSION_SHORT}}"
            LONG_VERSION = __long_version__ = "{long_version}.{{VERSION}}"
            {global_var_prefix}_CODEGEN_BRANCH = "{gitversion.git_current_branch()}"
            {global_var_prefix}_CODEGEN_VERSION = "{gitversion.version(append_hash=True,append_date=True)}"
            {global_var_prefix}_CODEGEN_REV = "{gitversion.git_rev()}"
            {global_var_prefix}_BRANCH = "{{BRANCH}}"
            {global_var_prefix}_VERSION = "{{VERSION}}"
            {global_var_prefix}_REV = "{{REV}}"

            {global_var_prefix}_MODULE_LIST = [
            {module_list_as_str}
            ]
            """
            ).replace("\t", "  ")
        )


def write_rocm_package_version_py_in(
    main_dir: str,
    global_var_prefix: str,  # e.g. ROCM_LLVM_PYTHON
    author: str,  # e.g. "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
    license_text: str,
    rocm_version: tuple,
    global_module_names: list,
):
    (_, _, version, long_version) = versions(*rocm_version)
    write_version_py_in(
        main_dir=main_dir,
        global_var_prefix=global_var_prefix,
        author=author,
        global_module_names=global_module_names,
        license_text=license_text,
        long_version=long_version,
        version=version,
    )


def write_subpackage_init_files(
    output_dir: str,
    root: it.Root,
    license_text: str,
    author: str,
):
    for dir in root.walk_directories():
        assert isinstance(dir, it.Directory)
        pkg_abspath = os.path.join(output_dir, dir.py_global_path)
        Path(pkg_abspath).mkdir(parents=False, exist_ok=True)
        with open(os.path.join(pkg_abspath, "__init__.py"), "w") as f:
            f.write(
                dir.py_render_init_file(
                    license_text=license_text,
                    author=author,
                )
            )


def write_autogenerated_module_files(
    output_dir: str, root: it.Root, libs_user_spec: str
):
    # process and check user-provided library names
    avail_lib_names = [node.py_name for node in root.walk_files()]
    chosen_lib_names = user_specified_lib_names(avail_lib_names, libs_user_spec)

    selected_global_module_names = []
    for libname in chosen_lib_names:
        for node in root.find_nodes(py_name=libname):
            if isinstance(node, it.File):
                parent_pkg_abspath = os.path.join(
                    output_dir, node.parent.py_global_path
                )
                selected_global_module_names.append(node.py_global_name)
                node.codegen.write_module_files(output_dir=parent_pkg_abspath)
    return selected_global_module_names


def generate_all_rocm_package_files(
    root: it.Root,
    pkg_opts: RocmPackageOpts,
    main_dir: str = None,
    main_child_modules: list = None,
    main_init_file_epilog: str = "",
):
    """
    Args:
        main_dir (str,optional):
            Location to write the package's main init and version file.
            If not specified or set to ``None``, a combination of ``pkgs_opts.package_dir``
            and the specified Python namespace is used. Defaults to ``None``.
        main_init_file_epilog (str,optional):
            Epilog to append to the main dir's init file. Defaults to ``""``.)
        main_child_modules (list(str), optional):
            Names of the child module of the main module.
            If an empty list is passed, no imports are generated.
            If ``None`` is specified, the root trees child modules are
            used. Defaults to ``None``.
    """
    # create directories
    output_dir = pkg_opts.package_dir
    root.py_create_package_dirs(output_dir, output_dir_parents=True)

    # render subdirectory init files
    write_subpackage_init_files(
        output_dir=output_dir,
        root=root,
        license_text=MIT_LICENSE_AMD,
        author=pkg_opts.author_email,
    )

    global_module_names = write_autogenerated_module_files(
        output_dir=output_dir, root=root, libs_user_spec=pkg_opts.libs_user_spec
    )

    # create init and version file in main dir
    if not main_dir:
        main_dir = os.path.join(output_dir, root.py_global_path)
    if (
        main_child_modules == None
    ):  # ! NOTE: we do not use 'if not main_child_modules:' by purpose !
        main_child_modules = sorted(
            set([f"{c.py_name}" for c in root.children])
        )  # removes duplicates due to splits etc.
    write_rocm_package_main_init_file(
        main_dir,
        pkg_opts.author_email,
        pkg_opts.rocm_version,
        main_child_modules,
        epilog=main_init_file_epilog,
    )
    write_rocm_package_version_py_in(
        main_dir=main_dir,
        global_var_prefix=pkg_opts.env_var_prefix,
        author=pkg_opts.author_email,
        license_text=MIT_LICENSE_AMD,
        rocm_version=pkg_opts.rocm_version,
        global_module_names=global_module_names,
    )
