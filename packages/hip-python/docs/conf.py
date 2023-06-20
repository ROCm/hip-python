# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# requires: pip install rocm-docs myst-parser

import os
import sys

# NOTE: always install the HIP Python packages, do not add the source folders
# to the sys path, i.e. do not add .. and ../hip-python-as-cuda!
# This breaks autodoc's automodule routine.

extensions = [
  "sphinx.ext.autodoc",  # Parses docstrings
  "myst_parser", # Allows to embed reST code in Markdown code
]

# TODO only for debugging
exclude_patterns = [ "python_api/" ]

project = 'HIP Python'
copyright = '2023, AMD_AUTHOR'
author = 'AMD_AUTHOR'

default_role = "py:obj" # this means that `test` will be expanded to :py:obj`test`

# From rocm-docs-core

external_projects_remote_repository = ""

external_projects_current_project = project

from rocm_docs import ROCmDocs

docs_core = ROCmDocs(project)
#docs_core.run_doxygen()  # Only if Doxygen is required for this project
#docs_core.enable_api_reference()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)