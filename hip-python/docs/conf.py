# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# requires: pip install rocm-docs myst-parser

from rocm_docs import ROCmDocs


project = "HIP Python"
author = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc."
# article info
os_support = ["linux"]
date = "2023-06-23"

default_role = "py:obj"  # this means that `test` will be expanded to :py:obj`test`

# NOTE: always install the HIP Python packages, do not add the source folders
# to the sys path, i.e. do not add .. and ../hip-python-as-cuda as
# this breaks autodoc's automodule routine.

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__, __getitem__",
    "inherited-members": False,
    "show-inheritance": False,
    "imported-members": False,
    "member-order": "bysource",  # bysource: seems unfortunately not to work for Cython modules
}

# Rocm-docs-core
external_projects_remote_repository = ""
external_projects_current_project = "hip"

article_pages = [
    {
        "file": "index",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "1 min read",
    },
    {
        "file": "user_guide/0_install",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "5 min read",
    },
    {
        "file": "user_guide/1_usage",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "60 min read",
    },
    {
        "file": "user_guide/2_cuda_python_interop",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "20 min read",
    },
    {
        "file": "user_guide/3_datatypes",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "10 min read",
    },
    {
        "file": "user_guide/4_report_bugs",
        "os": os_support,
        "author": author,
        "date": date,
        "read-time": "20 min read",
    },
]

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(project)
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

extensions += [
    "sphinx.ext.autodoc",  # Automatically create API documentation from Python docstrings
]
