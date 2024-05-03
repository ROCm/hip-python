# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

"""Mappings from `clang.cindex.Type` to `numba.core.types` and `llvmlite.ir` types.
"""

import rocm.clang.cindex as ci

from llvmlite import ir

from numba.core import types

from . import cparser

_clang_to_numba_core_map = {
    ci.TypeKind.POINTER: types.voidptr,
    ci.TypeKind.INCOMPLETEARRAY: types.voidptr,  # []
    ci.TypeKind.VOID: types.void,
    ci.TypeKind.BOOL: types.boolean,
    ci.TypeKind.CHAR_U: types.uchar,
    ci.TypeKind.UCHAR: types.uchar,
    ci.TypeKind.CHAR16: None,
    ci.TypeKind.CHAR32: None,
    ci.TypeKind.CHAR_S: types.char,
    ci.TypeKind.SCHAR: types.char,
    ci.TypeKind.WCHAR: None,
    ci.TypeKind.USHORT: types.uint16,
    ci.TypeKind.UINT: types.uint32,
    ci.TypeKind.ULONG: types.uint64,
    ci.TypeKind.ULONGLONG: types.uint64,
    ci.TypeKind.UINT128: None,
    ci.TypeKind.SHORT: types.int16,
    ci.TypeKind.INT: types.int32,
    ci.TypeKind.LONG: types.int64,
    ci.TypeKind.LONGLONG: types.int64,
    ci.TypeKind.INT128: None,
    ci.TypeKind.HALF: types.float16,
    ci.TypeKind.FLOAT: types.float32,
    ci.TypeKind.DOUBLE: types.float64,
    ci.TypeKind.LONGDOUBLE: None,
    ci.TypeKind.FLOAT128: None,
    ci.TypeKind.IBM128: None,
    (ci.TypeKind.COMPLEX, ci.TypeKind.FLOAT): types.complex64,
    (ci.TypeKind.COMPLEX, ci.TypeKind.DOUBLE): types.complex128,
}

# bools, chars: LLVM expresses signedness via operators, bool/char is i8.
# integer types: see LP64 datamodel https://en.cppreference.com/w/cpp/language/types
# pointer: chosen like generic pointer in numba/core/base.py
# complex: chosen as
_clang_to_llvmlite_map = {
    ci.TypeKind.POINTER: ir.PointerType(ir.IntType(8)),
    ci.TypeKind.INCOMPLETEARRAY: ir.PointerType(ir.IntType(8)),  # []
    ci.TypeKind.VOID: ir.VoidType(),
    ci.TypeKind.BOOL: ir.IntType(8),
    ci.TypeKind.CHAR_U: ir.IntType(8),
    ci.TypeKind.UCHAR: ir.IntType(8),
    ci.TypeKind.CHAR16: ir.IntType(16),
    ci.TypeKind.CHAR32: ir.IntType(32),
    ci.TypeKind.CHAR_S: ir.IntType(8),
    ci.TypeKind.SCHAR: ir.IntType(8),
    ci.TypeKind.WCHAR: ir.IntType(16),
    ci.TypeKind.USHORT: ir.IntType(16),
    ci.TypeKind.UINT: ir.IntType(32),
    ci.TypeKind.ULONG: ir.IntType(64),
    ci.TypeKind.ULONGLONG: ir.IntType(64),
    ci.TypeKind.UINT128: ir.IntType(128),
    ci.TypeKind.SHORT: ir.IntType(16),
    ci.TypeKind.INT: ir.IntType(32),
    ci.TypeKind.LONG: ir.IntType(64),
    ci.TypeKind.LONGLONG: ir.IntType(64),
    ci.TypeKind.INT128: ir.IntType(128),
    ci.TypeKind.HALF: ir.HalfType(),
    ci.TypeKind.FLOAT: ir.FloatType(),
    ci.TypeKind.DOUBLE: ir.DoubleType(),
    ci.TypeKind.LONGDOUBLE: None,
    ci.TypeKind.FLOAT128: None,
    ci.TypeKind.IBM128: None,
    (ci.TypeKind.COMPLEX, ci.TypeKind.FLOAT): ir.VectorType(ir.FloatType(), 2),
    (ci.TypeKind.COMPLEX, ci.TypeKind.DOUBLE): ir.VectorType(ir.DoubleType(), 2),
}

def map_clang_to_numba_core_type(clang_type: ci.Type):
    """Maps a Clang Python binding type to a ``numba.core`` equivalent."""
    global _clang_to_numba_core_map
    layers = tuple(
        cparser.TypeHandler.get(clang_type).walk_clang_type_layers(canonical=True)
    )
    if cparser.clang_type_kind(layers[0]) == ci.TypeKind.COMPLEX:
        type_map_arg = (cparser.clang_type_kind(layers[0]),cparser.clang_type_kind(layers[1]))
    else:
        type_map_arg = cparser.clang_type_kind(layers[0])
    numba_type = _clang_to_numba_core_map.get(type_map_arg, None)
    if numba_type:
        return numba_type
    elif cparser.clang_type_kind(clang_type) == ci.TypeKind.ENUM:
        return _clang_to_numba_core_map(clang_type.enum_type)
    elif cparser.clang_type_kind(clang_type) in (ci.TypeKind.RECORD, ci.TypeKind.CONSTANTARRAY):
        # TODO implement
        return None  # implies that it is ignored
    else:
        raise KeyError(
            f"clang type '{clang_type.spelling}' could not be mapped to a Numba type"
        )

def map_clang_to_llvmlite_type(clang_type: ci.Type):
    """Maps a Clang Python binding type to a ``numba.core`` equivalent."""
    global _clang_to_llvmlite_map
    layers = tuple(
        cparser.TypeHandler.get(clang_type).walk_clang_type_layers(canonical=True)
    )
    if cparser.clang_type_kind(layers[0]) == ci.TypeKind.COMPLEX:
        type_map_arg = (cparser.clang_type_kind(layers[0]),cparser.clang_type_kind(layers[1]))
    else:
        type_map_arg = cparser.clang_type_kind(layers[0])
    numba_type = _clang_to_llvmlite_map.get(type_map_arg, None)
    if numba_type:
        return numba_type
    elif cparser.clang_type_kind(clang_type) == ci.TypeKind.ENUM:
        return _clang_to_llvmlite_map(clang_type.enum_type)
    elif cparser.clang_type_kind(clang_type) in (ci.TypeKind.RECORD, ci.TypeKind.CONSTANTARRAY):
        # TODO implement
        return None  # implies that it is ignored
    else:
        raise KeyError(
            f"clang type '{clang_type.spelling}' could not be mapped to a Numba type"
        )
