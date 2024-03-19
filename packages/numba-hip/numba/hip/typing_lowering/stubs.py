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

import functools

from numba.core import types


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
    def has_attributes(cls):
        children = list(cls.get_children())
        typed_attributes = getattr(cls, "_typed_attributes_", {})
        return len(children) or len(typed_attributes)

    @classmethod
    def is_supported(cls):
        """If a Numba signature could be derived for the stub itself or any child.

        Note:
            Also returns 'False' if there is no child at all.
        """
        if hasattr(cls, "_signatures_") and len(cls._signatures_) > 0:
            return True
        elif cls.has_attributes():
            return True
        return False

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


class StubResolveAlways(Stub):
    """
    A stub that is always resolved as attribute
    of its parent even if it has no children
    and no typed attributes.
    """

    @classmethod
    def has_attributes(cls):
        return True


def stub_function(fn):
    """
    A stub function to represent special functions that are meaningless
    outside the context of a HIP kernel
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raise NotImplementedError("%s cannot be called from host code" % fn)

    return wrapped


# ------------------------------------------------------------------------------
# Attribute resolution
def resolve_attributes(
    registry, thekey: Stub, stub_attributes: dict, typed_attributes: dict = {}
):
    """Tells Numba how to resolve attributes of Python object 'thekey' (recursively)

    This function walks through the whole stub hierarchy and calls itself for
    every attribute that is a parent stub (stub without signatures but with children).
    For such parent stubs, the children stubs are passed as ``stub_attributes`` argument
    and class member `_typed_attributes_` is passed as ``typed_attributes`` argument.

    Example:
        Lets Numba know what an expression such as 'hip.syncthreads()' means
        given the object `hip`.

    Args:
        stub_attributes (`dict`):
            A dictionary that stores per attribute name key, a Stub instance <stub>.
            If the Stub instance has child stubs, the attribute is resolved as `numba.core.types.Module(<stub>)`
            If not, it is resolved as `numba.core.types.Function(<stub>)`
        typed_attributes (`dict`, optional):
            A dictionary that stores per attribute name key, an instance of `numba.core.types.Type`.
    Note:
        ``typed_attributes`` are checked first, which allows to overwrite
        (potentially automatically generated) stub attributes.
    See:
        `numba.core.typing.templates.AttributeTemplate._resolve`
    """
    from numba.core.typing.templates import AttributeTemplate

    assert isinstance(stub_attributes, dict)
    assert isinstance(typed_attributes, dict)

    @registry.register_attr
    class AttributeTemplate_(AttributeTemplate):
        key = types.Module(thekey)
        _stub_attributes: dict = stub_attributes
        _typed_attributes: dict = typed_attributes

        def __getattribute__(self, name: str):
            if name.startswith("resolve_"):
                attr = name.replace("resolve_", "")
                # 1. check typed attributes
                numba_type = self._typed_attributes.get(attr, None)
                if numba_type:
                    return lambda value: numba_type
                # 2. check stub attributes
                childstub: Stub = self._stub_attributes.get(attr, None)
                if childstub != None:
                    if childstub.is_supported():
                        if childstub.has_attributes():
                            assert not hasattr(
                                childstub, "_signatures_"
                            ), "function must not have children itself"
                            return lambda value: types.Module(
                                childstub
                            )  # register stub for parent stubs
                        else:
                            return lambda value: types.Function(
                                childstub._template_
                            )  # register concrete/callable template for function stubs
            return super().__getattribute__(name)

    for _, stub in stub_attributes.items():
        assert issubclass(stub, Stub)
        if stub.has_attributes():
            children = dict(stub.get_children())
            typed_attributes = getattr(stub, "_typed_attributes_", {})
            resolve_attributes(registry, stub, children, typed_attributes)
