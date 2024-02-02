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

import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter


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


def stub_function(fn):
    """
    A stub function to represent special functions that are meaningless
    outside the context of a HIP kernel
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        raise NotImplementedError("%s cannot be called from host code" % fn)

    return wrapped


# -------------------------------------------------------------------------------
# Array creation


class shared(Stub):
    """Shared memory namespace"""

    _description_ = "<shared>"

    @stub_function
    def array(shape, dtype):
        """Allocate a shared memory array

        Allocate a shared array of the given *shape* and *type*. *shape* is
        either an integer or a tuple of integers representing the array's
        dimensions.  *type* is a :ref:`Numba type <numba-types>` of the
        elements needing to be stored in the array.

        The returned array-like object can be read and written to like any
        normal device array (e.g. through indexing).

        Example usage:

        ```python
        sA = hip.shared.array(shape=(32, 32), dtype=float32)
        ```
        """


class local(Stub):
    """Local memory namespace"""

    _description_ = "<local>"

    @stub_function
    def array(shape, dtype):
        """Allocate a global memory array that is restricted to the current thread.

        Allocate a local array of the given *shape* and *type*. The array is
        private to the current thread, and resides in global memory. An
        array-like object is returned which can be read and written to like any
        standard array (e.g.  through indexing).

        Example usage:

        ```python
        lA = hip.local.array(shape=(4, 4), dtype=float32)
        ```
        """


class const(Stub):
    """Constant memory namespace"""

    @stub_function
    def array_like(ndarray):
        """Clone the other array

        Create a const array from *ndarry*. The resulting const array will have
        the same shape, type, and values as *ndarray*.
        """


# -------------------------------------------------------------------------------
# Cooperative groups

# TODO Not supported yet.


# -------------------------------------------------------------------------------
# vector types

# TODO Directly generate these simple types from the HIPRTC Header file.
# Needed to copy the Numba CUDA code as the vector-type base type ("Stub")
# cannot be supplied as parameter to make_vector_type_stubs. Furthermore, didn't
# want to trigger any init code of numba.cuda via an import statement.


def make_vector_type_stubs():
    """Make user facing objects for vector types"""
    vector_type_stubs = []
    vector_type_prefix = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    )
    vector_type_element_counts = (1, 2, 3, 4)
    vector_type_attribute_names = ("x", "y", "z", "w")

    for prefix, nelem in itertools.product(
        vector_type_prefix, vector_type_element_counts
    ):
        type_name = f"{prefix}x{nelem}"
        attr_names = vector_type_attribute_names[:nelem]

        vector_type_stub = type(
            type_name,
            (Stub,),  #:
            {
                **{attr: lambda self: None for attr in attr_names},
                **{
                    "_description_": f"<{type_name}>",
                    "__signature__": Signature(
                        parameters=[
                            Parameter(name=attr_name, kind=Parameter.POSITIONAL_ONLY)
                            for attr_name in attr_names[:nelem]
                        ]
                    ),
                    "__doc__": f"A stub for {type_name} to be used in " "HIP kernels.",
                },
                **{"aliases": []},
            },
        )
        vector_type_stubs.append(vector_type_stub)
    return vector_type_stubs


def map_vector_type_stubs_to_alias(vector_type_stubs):
    """For each of the stubs, create its aliases.

    For example: float64x3 -> double3
    """
    # C-compatible type mapping, see:
    # https://numpy.org/devdocs/reference/arrays.scalars.html#integer-types
    base_type_to_alias = {
        "char": f"int{np.dtype(np.byte).itemsize * 8}",
        "short": f"int{np.dtype(np.short).itemsize * 8}",
        "int": f"int{np.dtype(np.intc).itemsize * 8}",
        "long": f"int{np.dtype(np.int_).itemsize * 8}",
        "longlong": f"int{np.dtype(np.longlong).itemsize * 8}",
        "uchar": f"uint{np.dtype(np.ubyte).itemsize * 8}",
        "ushort": f"uint{np.dtype(np.ushort).itemsize * 8}",
        "uint": f"uint{np.dtype(np.uintc).itemsize * 8}",
        "ulong": f"uint{np.dtype(np.uint).itemsize * 8}",
        "ulonglong": f"uint{np.dtype(np.ulonglong).itemsize * 8}",
        "float": f"float{np.dtype(np.single).itemsize * 8}",
        "double": f"float{np.dtype(np.double).itemsize * 8}",
    }

    base_type_to_vector_type = defaultdict(list)
    for stub in vector_type_stubs:
        base_type_to_vector_type[stub.__name__[:-2]].append(stub)

    for alias, base_type in base_type_to_alias.items():
        vector_type_stubs = base_type_to_vector_type[base_type]
        for stub in vector_type_stubs:
            nelem = stub.__name__[-1]
            stub.aliases.append(f"{alias}{nelem}")


_vector_type_stubs = make_vector_type_stubs()
map_vector_type_stubs_to_alias(_vector_type_stubs)
