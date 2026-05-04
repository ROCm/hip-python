# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime as _datetime

_today = _datetime.today()

# Rocm-docs-core
external_projects_remote_repository = ""
external_projects = ["python", "rocm"]
external_projects_current_project = "hip-python"

setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = (
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
)
all_article_info_date = _today.strftime(r"%Y-%m-%d")

# specific settings override any general settings (eg: all_article_info_<field>)
article_pages = [
    {
        "file": "index",
        "read-time": "1 min read",
    },
    {
        "file": "user_guide/0_install",
        "read-time": "5 min read",
    },
    {
        "file": "user_guide/1_usage",
        "read-time": "60 min read",
    },
    {
        "file": "user_guide/2_cuda_python_interop",
        "read-time": "20 min read",
    },
    {
        "file": "user_guide/3_datatypes",
        "read-time": "10 min read",
    },
    {
        "file": "user_guide/4_report_bugs",
        "read-time": "20 min read",
    },
]

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

external_toc_path = "./sphinx/_toc.yml"

extensions = [
    "rocm_docs",
    "sphinx.ext.autodoc",  # Automatically create API documentation from Python docstrings
]


project = "HIP Python"
author = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
copyright = f"Copyright (c) 2023-{_today.strftime(r'%Y')} Advanced Micro Devices, Inc. All rights reserved."

default_role = (
    "py:obj"  # this means that `test` will be expanded to :py:obj`test`
)

# NOTE: always install the HIP Python packages, do not add the source folders
# to the sys path, i.e. do not add .. and ../hip-python-as-cuda as
# this breaks autodoc's automodule routine.

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__, __getitem__",
    "inherited-members": True,
    "show-inheritance": True,
    "imported-members": False,
    "member-order": "bysource",  # bysource: seems unfortunately not to work for Cython modules
}
