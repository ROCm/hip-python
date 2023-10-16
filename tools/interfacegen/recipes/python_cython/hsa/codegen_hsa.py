# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

"""This is the project's setup script.

After pointing this setup script to an HSA installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "Advanced Micro Devices, Inc. <rocm-hsa.maintainer@amd.com>"

import os
import warnings
import enum
import textwrap
import argparse

# configure warnings
original_formatwarning = warnings.formatwarning
def custom_formatwarning(warnobj,*args,**kwargs):
    global original_formatwarning
    if isinstance(warnobj,UserWarning):
        return f"Warning: {str(warnobj)}\n"
    else:
        return original_formatwarning(warnobj,*args,**kwargs)
warnings.formatwarning = custom_formatwarning

import interfacegen.gitversion
import interfacegen.cython

# configure codegen
# see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#role-py-obj
interfacegen.cython.python_interface_pyobj_role_template = r"`~.{name}`" # ~: removes the qualifier from the link text

from interfacegen.cython import (
    CythonPackageGenerator,
    DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

from interfacegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from interfacegen.tree import (
    Node,
    MacroDefinition,
    Parm,
)

def parse_options():
    global OUTPUT_DIR
    global ROCM_HSA_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global LIBS

    def get_bool_environ_var(env_var, default):
        yes_vals = ("true", "1", "t", "y", "yes")
        no_vals = ("false", "0", "f", "n", "no")
        value = os.environ.get(env_var, default).lower()
        if value in yes_vals:
            return True
        elif value in no_vals:
            return False
        else:
            allowed_vals = ", ".join(
                [f"'{a}'" for a in (list(yes_vals) + list(no_vals))]
            )
            raise RuntimeError(
                f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}"
            )

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
        Generator for ROCm HSA Python packages 'rocm-hsa'.
    
        NOTE:
            You can also use the environment variables 'ROCM_PATH' (or 'ROCM_HOME'),
            'ROCM_HSA_CLANG_RES_DIR', 'ROCM_HSA_LIBS',
            'ROCM_HSA_RUNTIME_LINKING' instead of the command line interface.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def dir_path(arg):
        if not os.path.isdir(arg):
            raise NotADirectoryError(arg)
        return arg

    parser.add_argument(
        "output_dir",
        type=dir_path,
        help="The output directory to which the files should be written to. Must contain `rocm-hsa` subfolder.",
    )
    parser.add_argument(
        "--rocm-path",
        type=str,
        required=False,
        dest="rocm_path",
        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.",
    )
    parser.add_argument(
        "--clang-resource-dir",
        required=False,
        dest="clang_resource_dir",
        help="The clang resource directory. Can also be set via environment variable 'ROCM_HSA_CLANG_RES_DIR'.",
    )
    parser.add_argument(
        "--libs",
        type=str,
        required=False,
        dest="libs",
        help="The ROCm libaries to generate interfaces for, as comma-separated list, e.g. 'hip,hiprtc'. Pass '*' to generate all, pass '' to generate none. Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.",
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
        clang_resource_dir=os.environ.get("ROCM_HSA_CLANG_RES_DIR", None),
        libs=os.environ.get("ROCM_HSA_LIBS", "*"),
        runtime_linking=get_bool_environ_var("ROCM_HSA_RUNTIME_LINKING", "true"),
        verbose=False,
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    RUNTIME_LINKING = args.runtime_linking
    LIBS = args.libs

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_HSA_INC = os.path.join(args.rocm_path,"include")

    GENERATOR_ARGS = [f"-I{ROCM_HSA_INC}"]
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
    GENERATOR_ARGS += ["-resource-dir", args.clang_resource_dir]


def generate_hsa_package_files():
    global OUTPUT_DIR
    global ROCM_HSA_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS

    global ROCM_HSA_VERSION_MAJOR
    global ROCM_HSA_VERSION_MINOR
    global ROCM_HSA_VERSION_PATCH
    global ROCM_HSA_VERSION_GITHASH

    def toclassname(name: str):
        return name[0].upper() + name[1:]

    def hsa_ptr_complicated_type_handler(parm: Node):
        return DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonPackageGenerator(
        "hsa",
        ROCM_HSA_INC,
        "hsa/hsa_ext_amd.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhsa-runtime64.so",
        #node_filter=_controls.hip.node_filter,
        #ptr_parm_intent=_controls.hip.ptr_parm_intent,
        #ptr_rank=_controls.hip.ptr_rank,
        #ptr_complicated_type_handler=hsa_ptr_complicated_type_handler,
        #macro_type=_controls.hip.macro_type,
        cflags=GENERATOR_ARGS,
    )
    ROCM_HSA_VERSION_MAJOR = 0
    ROCM_HSA_VERSION_MINOR = 0
    ROCM_HSA_VERSION_PATCH = 0
    ROCM_HSA_VERSION_GITHASH = ""
    #for node in generator.backend.root.walk():
    #    if isinstance(node, MacroDefinition):
    #        last_token = list(node.cursor.get_tokens())[-1].spelling
    #        if node.name == "ROCM_HSA_VERSION_MAJOR":
    #            ROCM_HSA_VERSION_MAJOR = int(last_token)
    #        elif node.name == "ROCM_HSA_VERSION_MINOR":
    #            ROCM_HSA_VERSION_MINOR = int(last_token)
    #        elif node.name == "ROCM_HSA_VERSION_PATCH":
    #            ROCM_HSA_VERSION_PATCH = int(last_token)
    #        elif node.name == "ROCM_HSA_VERSION_GITHASH":
    #            ROCM_HSA_VERSION_GITHASH = last_token.strip('"')
    return generator


if __name__ == "__main__":
    OUTPUT_DIR = None

    ROCM_HSA_INC = None
    RUNTIME_LINKING = None
    GENERATOR_ARGS = None
    LIBS = None

    ROCM_HSA_VERSION_MAJOR = 0
    ROCM_HSA_VERSION_MINOR = 0
    ROCM_HSA_VERSION_PATCH = 0
    ROCM_HSA_VERSION_GITHASH = ""

    parse_options()

    AVAILABLE_GENERATORS = dict(
        hsa=generate_hsa_package_files,
    )

    # process and check user-provided library names
    avail_lib_names = AVAILABLE_GENERATORS.keys()
    processed_libs = LIBS.replace(" ","")
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
                raise ValueError(f"library name '{name}' is not valid, use one of: {', '.join(avail_lib_names)}")

    hsa_output_dir = os.path.join(OUTPUT_DIR, "rocm", "hsa")
    for entry in avail_lib_names:
        libname = entry.strip()
        if libname not in AVAILABLE_GENERATORS:
            available_libs = ", ".join([f"'{a}'" for a in AVAILABLE_GENERATORS.keys()])
            msg = f"no codegenerator found for library '{libname}'; please choose from: {available_libs}, or '*', which implies that all code generators will be used."
            raise KeyError(msg)
        generator = AVAILABLE_GENERATORS[libname]()
        generator.write_package_files(output_dir=hsa_output_dir)

    ROCM_HSA_VERSION_NAME = f"{ROCM_HSA_VERSION_MAJOR}.{ROCM_HSA_VERSION_MINOR}.{ROCM_HSA_VERSION_PATCH}-{ROCM_HSA_VERSION_GITHASH}"
    ROCM_HSA_VERSION = (
        ROCM_HSA_VERSION_MAJOR * 10000000 + ROCM_HSA_VERSION_MINOR * 100000 + ROCM_HSA_VERSION_PATCH
    )

    VERSION = f"{ROCM_HSA_VERSION_MAJOR}.{ROCM_HSA_VERSION_MINOR}.{ROCM_HSA_VERSION_PATCH}.{interfacegen.gitversion.version()}"
    LONG_VERSION = (
        f"{ROCM_HSA_VERSION_NAME}.{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
    )
    
    with open(os.path.join("..","LICENSE"),"r") as licensefile:
        LICENSE_TEXT = "".join([f"# {ln}\n" for ln in licensefile.read().rstrip().splitlines()])
    # rocm/hsa/_version.py
    with open(os.path.join(hsa_output_dir, "_version.py.in"), "w") as f:
        f.write(
            LICENSE_TEXT
            + textwrap.dedent(
            f"""\
            
            # This file has been autogenerated, do not modify.
            
            __author__ = "Advanced Micro Devices, Inc. <rocm-hsa.maintainer@amd.com>"

            VERSION = __version__ = "{VERSION}.{{ROCM_HSA_VERSION_SHORT}}"
            LONG_VERSION = __long_version__ = "{LONG_VERSION}.{{ROCM_HSA_VERSION}}"
            ROCM_HSA_CODEGEN_BRANCH = "{interfacegen.gitversion.git_current_branch()}"
            ROCM_HSA_CODEGEN_VERSION = "{interfacegen.gitversion.version(append_hash=True,append_date=True)}"
            ROCM_HSA_CODEGEN_REV = "{interfacegen.gitversion.git_rev()}"
            ROCM_HSA_BRANCH = "{{ROCM_HSA_BRANCH}}"
            ROCM_HSA_VERSION = "{{ROCM_HSA_VERSION}}"
            ROCM_HSA_REV = "{{ROCM_HSA_REV}}"\
            """
            ).strip()
        )
    # hsa/__init__.py
    # TODO make option to use all generators or only specified ones
    ROCM_HSA_LIB_NAMES = AVAILABLE_GENERATORS.keys()

    with open(os.path.join(hsa_output_dir, "__init__.py"), "w") as f:
        init_content = (
            LICENSE_TEXT
            + textwrap.dedent(
                f"""\
            
                # This file has been autogenerated, do not modify.
                
                __author__ = "Advanced Micro Devices, Inc. <rocm-hsa.maintainer@amd.com>"

                from ._version import *
                ROCM_HSA_VERSION = {ROCM_HSA_VERSION}
                ROCM_HSA_VERSION_NAME = hsa_version_name = "{ROCM_HSA_VERSION_NAME}"
                ROCM_HSA_VERSION_TUPLE = hsa_version_tuple = ({ROCM_HSA_VERSION_MAJOR},{ROCM_HSA_VERSION_MINOR},{ROCM_HSA_VERSION_PATCH},"{ROCM_HSA_VERSION_GITHASH}")

                """
            )
        )
        init_content += "\nfrom . import _util"
        for pkg_name in ROCM_HSA_LIB_NAMES:
            init_content += textwrap.dedent(f"""
            try:
                from . import {pkg_name}
            except ImportError:
                pass # may have been excluded from build""")
        f.write(init_content)
    # rocm-hsa docs
    # files per api

    #def write_pkg_markdown_file_(pkg,lib,extra=""):
    #    with open(os.path.join(ROCM_HSA_DOCS, "python_api", f"{lib}.md"),"w") as outfile:
    #        outfile.write(textwrap.dedent(
    #            f"""\
    #            # {pkg}.{lib}
    #            
    #            <!-- This file has been autogenerated, do not modify. -->

    #            <!-- global automodule options are set in conf.py -->
    #            ```{{eval-rst}}
    #            .. automodule:: {pkg}.{lib}
    #            {extra}

    #            ```"""
    #        ))

    #ROCM_HSA_DOCS = os.path.join(OUTPUT_DIR,"rocm-hsa","docs")
    #for lib in ROCM_HSA_LIB_NAMES:
    #    write_pkg_markdown_file_("hip",lib)
    ## index.md from index.md.in
    #index_md = os.path.join(
    #    ROCM_HSA_DOCS, "index.md"
    #)
    #PYTHON_API_DOC_NAMES = [f"- {{doc}}`python_api/{lib}`" for lib in ROCM_HSA_LIB_NAMES]
    #with open(index_md + ".in","r"
    #     ) as infile, open(index_md, "w") as outfile:
    #    
    #    for key in AVAILABLE_GENERATORS:
    #        rendered = infile.read()
    #        rendered = rendered.replace("{PYTHON_API_DOC_NAMES}","\n".join(PYTHON_API_DOC_NAMES))
    #        rendered = rendered.replace("{ROCM_HSA_VERSION_NAME}", ROCM_HSA_VERSION_NAME)
    #        outfile.write(rendered)
    ## _toc.yml.in from _toc.yml.in.in
    #toc_yml_md_in = os.path.join(
    #    ROCM_HSA_DOCS, ".sphinx", "_toc.yml.in"
    #)
    #PYTHON_API_FILE_NAMES = [f"      - file: python_api/{lib}" for lib in ROCM_HSA_LIB_NAMES]
    #with open(toc_yml_md_in + ".in","r"
    #     ) as infile, open(toc_yml_md_in, "w") as outfile:
    #    
    #    for key in AVAILABLE_GENERATORS:
    #        rendered = infile.read()
    #        rendered = rendered.replace("{PYTHON_API_FILE_NAMES}","\n".join(PYTHON_API_FILE_NAMES))
    #        outfile.write(rendered)
