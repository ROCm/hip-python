# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# requires: pip install rocm-docs myst-parser

import os
import sys

project = 'HIP Python'
copyright = 'Copyright (c) 2023 Advanced Micro Devices, Inc.'
author = 'Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>'

default_role = "py:obj" # this means that `test` will be expanded to :py:obj`test`

extensions = [
  "sphinx.ext.autodoc",  # Automatically create API documentation from Python docstrings
  "myst_parser", # Allows to embed reST code in Markdown code
]

# NOTE: always install the HIP Python packages, do not add the source folders
# to the sys path, i.e. do not add .. and ../hip-python-as-cuda as 
# this breaks autodoc's automodule routine.

autodoc_member_order = 'bysource' # Order members by source appearance

# Rocm-docs-core
from rocm_docs import ROCmDocs
external_projects_remote_repository = ""
external_projects_current_project = project

docs_core = ROCmDocs(project)
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
