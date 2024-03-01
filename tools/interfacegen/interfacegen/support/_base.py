
import os
import re
import datetime
import argparse
import textwrap

from . import gitversion

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

THIS_YEAR_YYYY = datetime.date.today().strftime("%Y")

COMMENT_CHAR_PY = "#"
COMMENT_CHAR_F90 = "!"

MIT_LICENSE_AMD = """\
{comment_char} MIT License
{comment_char}
{comment_char} Copyright (c) {copyright_year} Advanced Micro Devices, Inc.
{comment_char}
{comment_char} Permission is hereby granted, free of charge, to any person obtaining a copy
{comment_char} of this software and associated documentation files (the "Software"), to deal
{comment_char} in the Software without restriction, including without limitation the rights
{comment_char} to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
{comment_char} copies of the Software, and to permit persons to whom the Software is
{comment_char} furnished to do so, subject to the following conditions:
{comment_char}
{comment_char} The above copyright notice and this permission notice shall be included in all
{comment_char} copies or substantial portions of the Software.
{comment_char}
{comment_char} THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
{comment_char} IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
{comment_char} FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
{comment_char} AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
{comment_char} LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
{comment_char} OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
{comment_char} SOFTWARE.
"""

def render_license_MIT(
    year_end: str = THIS_YEAR_YYYY,
    year_start: str = THIS_YEAR_YYYY,
    comment_char=COMMENT_CHAR_PY,
):
    if year_end == year_start:
        copyright_year = year_end
    else:
        copyright_year = f"{year_start}-{year_end}"
    return MIT_LICENSE_AMD.format(
        copyright_year=copyright_year, comment_char=comment_char
    )


def versions(rocm_version_major: int, rocm_version_minor: int, rocm_version_patch: int):
    rocm_version_name: str = (
        f"{rocm_version_major}.{rocm_version_minor}.{rocm_version_patch}"
    )
    rocm_version: int = (
        rocm_version_major * 10000000 + rocm_version_minor * 100000 + rocm_version_patch
    )

    version: str = (
        f"{rocm_version_name}.{gitversion.git_branch_rev_count(gitversion.git_current_branch())}"
    )
    long_version: str = (
        f"{rocm_version_name}.{gitversion.version(append_hash=True,append_date=True)}"
    )
    return (rocm_version, rocm_version_name, version, long_version)

def render_license_MIT(
    year_end = THIS_YEAR_YYYY,
    year_start = THIS_YEAR_YYYY,
    comment_char=COMMENT_CHAR_PY,
):
    year_end = str(year_end)
    year_start = str(year_start)
    if year_end == year_start:
        copyright_year = year_end
    else:
        copyright_year = f"{year_start}-{year_end}"
    return MIT_LICENSE_AMD.format(
        copyright_year=copyright_year, comment_char=comment_char
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
        verbose=False,
    )
    return parser

class PackageOpts:
    def __init__(self):
        self.output_dir: str = None
        self.package_dir: str = None
        self.rel_inc_dir: str = None
        self.abs_inc_dir: str = None
        self.env_var_prefix: str = None
        self.clang_res_dir: str = None
        self.generator_args: list = None
        self.libs_user_spec: str = None
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
    def rocm_version_major(self):
        return self.rocm_version[0]

    @property
    def rocm_version_minor(self):
        return self.rocm_version[1]

    @property
    def rocm_version_patch(self):
        return self.rocm_version[2]


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


def lstrip_all_lines(text: str, lstrip_chars: str = " "):
    return "".join(
        [line.lstrip(lstrip_chars) for line in text.splitlines(keepends=True)]
    )

def _create_rocm_package_opts_from_cli(
    rocm_package_opts_type,
    project: str,
    env_var_prefix: str,
    libs_example: str,
    package: str,
    rel_inc_dir: str,
    author: str,
    email: str,
    dll: str = None,
    util_pkg: str = None,
    parser_builder = create_rocm_package_cli_parser,
):
    rocm_package_opts = rocm_package_opts_type()
    try:
        rocm_package_opts.dll = dll
        rocm_package_opts.util_pkg = util_pkg
    except AttributeError:
        pass
    rocm_package_opts.author = author
    rocm_package_opts.email = email
    rocm_package_opts.env_var_prefix = env_var_prefix

    parser = parser_builder(
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

    rocm_package_opts.output_dir = args.output_dir
    rocm_package_opts.package_dir = os.path.join(args.output_dir, package)
    try:
        rocm_package_opts.runtime_linking = args.runtime_linking
    except AttributeError:
        pass
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
    return rocm_package_opts