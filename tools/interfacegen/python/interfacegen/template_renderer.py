# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""In-memory rendering of CMake-style header templates (.h.in files).

Supports parsing version information from ROCm repository metadata files and
rendering templates without requiring CMake configure step.
"""

import os
import re
from typing import Dict


def parse_rccl_version(version_mk_path: str) -> Dict[str, int]:
    """Parse RCCL version from makefiles/version.mk.

    Args:
        version_mk_path: Path to rocm-systems/projects/rccl/makefiles/version.mk

    Returns:
        Dictionary with keys: NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_VERSION

    Raises:
        FileNotFoundError: If version file doesn't exist
        ValueError: If version parsing fails
    """
    if not os.path.exists(version_mk_path):
        raise FileNotFoundError(f"RCCL version file not found: {version_mk_path}")

    with open(version_mk_path, 'r') as f:
        content = f.read()

    # Parse Makefile-style assignments: NCCL_MAJOR := 2
    major_match = re.search(r'NCCL_MAJOR\s*:=\s*(\d+)', content)
    minor_match = re.search(r'NCCL_MINOR\s*:=\s*(\d+)', content)
    patch_match = re.search(r'NCCL_PATCH\s*:=\s*(\d+)', content)

    if not all([major_match, minor_match, patch_match]):
        raise ValueError(f"Failed to parse RCCL version from {version_mk_path}")

    major = int(major_match.group(1))
    minor = int(minor_match.group(1))
    patch = int(patch_match.group(1))

    return {
        "NCCL_MAJOR": major,
        "NCCL_MINOR": minor,
        "NCCL_PATCH": patch,
        "NCCL_VERSION": (major * 10000) + (minor * 100) + patch,
    }


def parse_comgr_version(version_txt_path: str) -> Dict[str, str]:
    """Parse COMGR version from VERSION.txt.

    Args:
        version_txt_path: Path to llvm-project/amd/comgr/VERSION.txt

    Returns:
        Dictionary with keys: AMD_COMGR_VERSION_MAJOR, AMD_COMGR_VERSION_MINOR, AMD_COMGR_VERSION_PATCH

    Raises:
        FileNotFoundError: If version file doesn't exist
        ValueError: If version format is invalid
    """
    if not os.path.exists(version_txt_path):
        raise FileNotFoundError(f"COMGR version file not found: {version_txt_path}")

    with open(version_txt_path, 'r') as f:
        version_str = f.read().strip()

    # Parse version string: "2.8.0"
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid COMGR version format: {version_str}")

    return {
        "AMD_COMGR_VERSION_MAJOR": parts[0],
        "AMD_COMGR_VERSION_MINOR": parts[1],
        "AMD_COMGR_VERSION_PATCH": parts[2],
    }


def render_template(template_path: str, variables: Dict[str, any]) -> str:
    """Render CMake-style template by substituting variables.

    Supports both ${VAR} and @VAR@ substitution syntax.

    Args:
        template_path: Path to .h.in template file
        variables: Dictionary of variable name -> value

    Returns:
        Rendered template content as string

    Raises:
        FileNotFoundError: If template doesn't exist
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        content = f.read()

    # Convert all values to strings
    str_vars = {k: str(v) for k, v in variables.items()}

    # Substitute ${VAR} and @VAR@ patterns
    for var_name, value in str_vars.items():
        content = content.replace(f"${{{var_name}}}", value)
        content = content.replace(f"@{var_name}@", value)

    return content
