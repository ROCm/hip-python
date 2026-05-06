# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR IN CONNECTION WITH THE
# SOFTWARE.

"""Version information for ROCm bindings, derived from package metadata.

This module reads version information from installed package metadata and provides
it in multiple convenient formats (raw string, tuple, integer).

Usage:
    from rocm import version

    print(version.ROCM_VERSION_TUPLE)  # (7, 13, 0)
    print(version.ROCM_VERSION_NAME)   # "7.13.0"
    print(version.ROCM_VERSION)        # 71300000
    print(version.HIP_VERSION_TUPLE)   # (7, 13, 26154, "92b7431876")

    # Or import specific attributes:
    from rocm.version import ROCM_VERSION_TUPLE, HIP_VERSION_TUPLE
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import sys
from typing import Tuple, Union, Optional, Dict, Any

# Try to import metadata functionality
if sys.version_info >= (3, 8):
    from importlib.metadata import version as get_version, metadata as get_metadata, PackageNotFoundError
else:
    try:
        from importlib_metadata import version as get_version, metadata as get_metadata, PackageNotFoundError
    except ImportError:
        # Fallback for very old Python without importlib_metadata
        def get_version(pkg):
            raise PackageNotFoundError(pkg)
        def get_metadata(pkg):
            raise PackageNotFoundError(pkg)
        PackageNotFoundError = Exception


def _get_tool_metadata(package_name: str) -> Optional[Dict[str, Any]]:
    """Get [tool.rocm-bindings] metadata from package if available.

    Note: This requires the metadata to be in PKG-INFO/METADATA, which is not
    standard for [tool.*] sections. May not work in all cases.

    Args:
        package_name: Name of package to query

    Returns:
        Dictionary of tool metadata if available, None otherwise
    """
    try:
        # Note: importlib.metadata doesn't directly expose [tool.*] sections
        # This is a limitation - tool sections are build-time only
        # We'll need to rely on parsing package version for ROCm version
        return None
    except Exception:
        return None


def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Parse version string into tuple of integers.

    Args:
        version_str: Version string like "7.13.0" or "7.13.26154"

    Returns:
        Tuple of version components as integers
    """
    parts = version_str.split('.')
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        # Handle non-numeric components
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(part)
        return tuple(result)


def _version_to_int(version_tuple: Tuple[int, ...]) -> int:
    """Convert version tuple to integer (e.g., (7, 13, 0) -> 71300000).

    Args:
        version_tuple: Version tuple (major, minor, patch)

    Returns:
        Integer version (major*10000000 + minor*100000 + patch)
    """
    major = version_tuple[0] if len(version_tuple) > 0 else 0
    minor = version_tuple[1] if len(version_tuple) > 1 else 0
    patch = version_tuple[2] if len(version_tuple) > 2 else 0
    return major * 10000000 + minor * 100000 + patch


# Get package version from rocm-bindings-core package metadata
try:
    __version__ = get_version("rocm-bindings-core")  # e.g., "7.13.0.563.61"

    # Extract ROCm version (first 3 parts) from package version
    _pkg_parts = __version__.split('.')
    if len(_pkg_parts) >= 3:
        # Package version is like "7.13.0.563.61" - take first 3 parts for ROCm version
        ROCM_VERSION_NAME = rocm_version_name = '.'.join(_pkg_parts[:3])  # "7.13.0"
        ROCM_VERSION_TUPLE = rocm_version_tuple = tuple(int(x) for x in _pkg_parts[:3])  # (7, 13, 0)
    else:
        # Fallback if version format unexpected
        ROCM_VERSION_NAME = rocm_version_name = __version__
        ROCM_VERSION_TUPLE = rocm_version_tuple = _parse_version_tuple(__version__)

    ROCM_VERSION = _version_to_int(ROCM_VERSION_TUPLE)

except PackageNotFoundError:
    # Fallback for development/editable installs
    __version__ = "7.13.0.563.61"
    ROCM_VERSION_NAME = rocm_version_name = "7.13.0"
    ROCM_VERSION_TUPLE = rocm_version_tuple = (7, 13, 0)
    ROCM_VERSION = 71300000


# Get HIP version
# Try multiple sources in order:
# 1. [tool.rocm-bindings] metadata (if available)
# 2. rocm-bindings-hip package version
# 3. Hardcoded fallback

_tool_meta = _get_tool_metadata("rocm-bindings-core")
if _tool_meta and "hip_full_version" in _tool_meta:
    # Got it from tool metadata
    HIP_VERSION_NAME = hip_version_name = _tool_meta["hip_full_version"]
else:
    # Try rocm-bindings-hip package metadata
    try:
        _hip_version = get_version("rocm-bindings-hip")

        # HIP package version might be like "7.13.26154.563.61"
        # Extract HIP version (first 4 parts if numeric, or full if has commit hash)
        _hip_parts = _hip_version.split('.')
        if len(_hip_parts) >= 4:
            # Reconstruct as "major.minor.patch-commit" if commit hash present
            hip_major_minor_patch = '.'.join(_hip_parts[:3])
            if len(_hip_parts) > 4:
                # Has additional parts - might include commit
                HIP_VERSION_NAME = hip_version_name = f"{hip_major_minor_patch}.{_hip_parts[3]}"
            else:
                HIP_VERSION_NAME = hip_version_name = '.'.join(_hip_parts[:4])
        else:
            HIP_VERSION_NAME = hip_version_name = _hip_version

    except PackageNotFoundError:
        # Fallback if rocm-bindings-hip not installed
        HIP_VERSION_NAME = hip_version_name = "7.13.26154-92b7431876"

# Parse HIP version tuple
# Format: "7.13.26154-92b7431876" → (7, 13, 26154, "92b7431876")
if '-' in HIP_VERSION_NAME:
    _ver_part, _commit = HIP_VERSION_NAME.split('-', 1)
    _nums = [int(x) for x in _ver_part.split('.')]
    HIP_VERSION_TUPLE = hip_version_tuple = tuple(_nums + [_commit])
else:
    # No commit hash, just numeric version
    _nums = [int(x) if x.isdigit() else x for x in HIP_VERSION_NAME.replace('-', '.').split('.')]
    HIP_VERSION_TUPLE = hip_version_tuple = tuple(_nums)

# Integer version uses first 3 numeric parts
_hip_numeric = [x for x in HIP_VERSION_TUPLE if isinstance(x, int)][:3]
HIP_VERSION = _version_to_int(tuple(_hip_numeric))


__all__ = [
    '__version__',
    'ROCM_VERSION',
    'ROCM_VERSION_NAME',
    'rocm_version_name',
    'ROCM_VERSION_TUPLE',
    'rocm_version_tuple',
    'HIP_VERSION',
    'HIP_VERSION_NAME',
    'hip_version_name',
    'HIP_VERSION_TUPLE',
    'hip_version_tuple',
]
