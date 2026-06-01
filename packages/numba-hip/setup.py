# MIT License
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Build customization for numba-hip.

All project metadata lives in ``pyproject.toml`` (PEP 621); this ``setup.py``
exists ONLY to override the ``bdist_wheel`` command so that the wheel is built
as a non-pure (platlib) wheel.

Why we force a non-pure (platlib) wheel even though numba-hip is currently
pure Python:

``numba`` ships compiled extensions, so pip classifies it as a *platlib*
package. On distros that split ``lib``/``lib64`` (e.g. RHEL/Fedora) platlib
resolves to ``.../lib64/.../site-packages`` while *purelib* resolves to
``.../lib/.../site-packages``. Because ``numba`` is a regular package (it ships
``numba/__init__.py``, not a namespace package), Python binds ``numba`` to a
single directory and never merges the other. A pure ``numba-hip`` wheel would
land in ``lib/`` and its ``numba/hip`` would be invisible to a ``numba`` rooted
in ``lib64/`` -- ``import numba.hip`` would then fail despite both being
installed.

Marking the wheel non-pure makes pip install it into the same (platlib) scheme
as ``numba``, so the two always co-locate. Once numba-hip grows real compiled
extensions this override becomes redundant (platlib is then automatic) and can
be removed.

How it is done:

* ``Distribution.has_ext_modules`` is forced to return ``True``. This is what
  routes the package files into the *platlib* install scheme (without it,
  setuptools places pure-Python packages under ``*.data/purelib/`` even when
  ``Root-Is-Purelib`` is false, which would still land them in ``lib/``).
* ``bdist_wheel.get_tag`` is overridden to keep the wheel interpreter-agnostic
  (``py3``/``none``) while retaining the platform tag -- otherwise marking the
  distribution non-pure would also make the tag interpreter-specific
  (e.g. ``cp312-cp312-...``). The build step then produces a
  ``py3-none-<platform>`` wheel; CI relabels the platform tag to manylinux.

Once numba-hip grows real compiled extensions this whole module becomes
redundant (platlib placement and tagging are then automatic) and can be removed.
"""

from setuptools import setup
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 vendors bdist_wheel (we require >= 77)
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - fallback for older toolchains
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Reports a non-pure distribution so files install into platlib.

    numba-hip has no compiled extensions today; this forces platlib placement
    so it co-locates with ``numba`` on lib/lib64-split systems.
    """

    def has_ext_modules(self):
        return True


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        # Keep the wheel interpreter-agnostic (py3-none) but platform-specific
        # (platlib). Without this, a non-pure dist yields a cpXY-specific tag.
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
