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

"""rocm.bindings.util - Utility types and loaders for ROCm bindings."""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# Version attributes are lazy-loaded from _version module
_dynamic_version_attrs = {
    "VERSION",
    "__version__",
    "LONG_VERSION",
    "__long_version__",
}

# Track if we're currently importing to prevent recursive lookups
_importing = set()


def __getattr__(name):
    """Lazy-load version attributes and submodules."""
    # Prevent recursive lookups during submodule initialization
    if name in _importing:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # Check if it's a version attribute
    if name in _dynamic_version_attrs:
        from . import _version

        value = getattr(_version, name)
        globals()[name] = value
        return value

    # Try to import as a submodule
    _importing.add(name)
    try:
        module = __import__(f"{__name__}.{name}", fromlist=[name])
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        # Not a submodule - it's an attribute that doesn't exist
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    finally:
        _importing.discard(name)
