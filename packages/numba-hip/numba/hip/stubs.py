# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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


class Stub(object):
    """Numba HIP stub object

    A stub object to represent special objects that are meaningless
    outside the context of a AMD GPU kernel.

    Numba typing signatures can be registered with this module.
    Numba call generators can be registered with this module.
    """

    _description_ = "<numba.hip special value>"
    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError(f"{cls} cannot be instantiated")

    def __repr__(self):
        return self._description_

    @classmethod
    def get_children(cls):
        """Yields all members of type `~.Stub`."""
        for name, child in vars(cls).items():
            try:
                if issubclass(child, Stub):
                    yield (name, child)
            except TypeError:
                pass

    @classmethod
    def is_supported(cls):
        """If a Numba signature could be derived for the stub itself or any child.

        Note:
            Also returns 'False' if there is no child at all.
        """
        if hasattr(cls, "_signatures_") and len(cls._signatures_) > 0:
            return True
        for _, child in cls.get_children():
            if child.is_supported():
                return True
        return False

    @classmethod
    def remove_unsupported(cls):
        """Remove children for which no Numba signature could be derived.

        Note:
            Works recursively if a child has children itself.
        """
        for name, child in cls.get_children():
            if not child.is_supported():
                delattr(cls, name)

    @classmethod
    def walk(cls, post_order=False):
        """Yield all stubs in this subtree.

        Args:
            post_order (bool, optional):
                Traverse in post-order, i.e. children are yielded before
                their parent. Defaults to False.

        Yields:
            Subclasses of `~.Stub`.
        """
        if not post_order:
            yield cls
        for (child,) in cls.get_children():
            yield from child.walk(cls, post_order)
        if post_order:
            yield cls
