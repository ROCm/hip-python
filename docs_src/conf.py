# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
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
        "file": "user_guide/3_jit_compilation",
        "read-time": "30 min read",
    },
    {
        "file": "user_guide/4_datatypes",
        "read-time": "10 min read",
    },
    {
        "file": "user_guide/5_report_bugs",
        "read-time": "20 min read",
    },
]

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

external_toc_path = "./sphinx/_toc.yml"

extensions = [
    "rocm_docs",
    # sphinx-autoapi parses Python (and .pyi stub) source files directly,
    # so the doc build does not need the compiled hip-python wheels on
    # `sys.path`. The generator emits .pyi stubs alongside every
    # generated .pxd/.pyx for the high-level Python API; each function
    # stub carries the parameter names and the .pyx docstring, so
    # autoapi has enough to render the API surface (see
    # share/design/CODEGEN.md).
    "autoapi.extension",
]


project = "HIP Python"
author = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"
copyright = f"Copyright (c) 2023-{_today.strftime(r'%Y')} Advanced Micro Devices, Inc. All rights reserved."

default_role = (
    "py:obj"  # this means that `test` will be expanded to :py:obj`test`
)

# ---------------------------------------------------------------------------
# sphinx-autoapi configuration
# ---------------------------------------------------------------------------
#
# autoapi parses each rocm/, cuda/, hip/ source tree directly. Combined with
# the generator-emitted .pyi stubs (per share/design/CODEGEN.md), this lets
# Sphinx render the API surface without importing any compiled extension.
#
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.normpath(os.path.join(_HERE, "..", "python"))

autoapi_type = "python"
autoapi_dirs = [
    os.path.join(_PYTHON_ROOT, "rocm-bindings-util", "rocm"),
    os.path.join(_PYTHON_ROOT, "rocm-bindings-hip", "rocm"),
    os.path.join(_PYTHON_ROOT, "rocm-bindings-libraries", "rocm"),
    os.path.join(_PYTHON_ROOT, "rocm-bindings-compiler", "rocm"),
    os.path.join(_PYTHON_ROOT, "hip-python-interop", "cuda"),
    os.path.join(_PYTHON_ROOT, "hip-python", "hip"),
]
autoapi_root = "python_api"
autoapi_keep_files = True
autoapi_add_toctree_entry = False  # the per-package _toc.yml manages TOC
autoapi_member_order = "bysource"
autoapi_python_use_implicit_namespaces = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
