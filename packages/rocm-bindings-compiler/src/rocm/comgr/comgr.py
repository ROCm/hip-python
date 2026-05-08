# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Higher level interfaces that simplify the use of AMD COMGR.

Attributes:
    HIPRTC_RUNTIME_HEADER (`str`):
        The content of the ``hipRTC`` / ``hiprtc_runtime/h`` header file.
        Take a look at https://github.com/ROCm/clr for more details on how this
        file is generated.
        The ``HIP`` and ``clr`` branches for generating the file have been
        selected according to ``rocm.bindings.comgr.ROCM_VERSION``.
"""

import ctypes
import os
import textwrap

import rocm.bindings.amd_comgr as _amd_comgr
from rocm.bindings.util.types import CStr
from rocm.version import ROCM_VERSION_TUPLE  # noqa: F401

from . import amd_hsa_kernel_descriptor


def to_bytes(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8")
    elif isinstance(obj, bytes):
        return obj
    else:
        return bytes(obj)


def to_str(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, bytes):
        return obj.decode()
    else:
        raise ValueError(
            f"input argument must of type 'str' or 'bytes'; is: {type(obj)}"
        )


def comgr_check(
    call_result,
):
    """Check AMD COMGR call status and return other result tuple entries."""
    if isinstance(call_result, tuple):
        err = call_result[0]
        result = call_result[1:]
    else:
        err = call_result
        result = None
    if result and len(result) == 1:
        result = result[0]
    if (
        isinstance(err, _amd_comgr.amd_comgr_status_s)
        and err != _amd_comgr.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


def metadata_string_get_bytes(
    metadata_string: _amd_comgr.amd_comgr_metadata_node_s,
):
    """Get the text associated with a string metadata node as `bytes`.

    Args:
        metadata_str (`~.amd_comgr_metadata_node_s`):
            A metadata node that represents a string.
    """
    # First determines length of string, then loads it into buffer.
    str_len = ctypes.c_ulong(0)
    comgr_check(
        _amd_comgr.amd_comgr_get_metadata_string(
            metadata_string, ctypes.addressof(str_len), None
        )
    )
    str_len.value -= 1  # remove the 0x00 terminator
    str_buf = bytes(str_len.value)
    comgr_check(
        _amd_comgr.amd_comgr_get_metadata_string(
            metadata_string, ctypes.addressof(str_len), str_buf
        )
    )
    return str_buf


@ctypes.CFUNCTYPE(None, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p)
def _get_map_keys_cb(ckey, _, userdata):
    """Callback for extracting keys from a metadata map.

    Intended to be used with ``amd_comgr_iterate_map_metadata``.

    Note:
        The callback has the following signature:

        ```cython
        void (*callback) (amd_comgr_metadata_node_s key,
          amd_comgr_metadata_node_s value, void * userdata)
        ```

        where

        ```cython
        cdef struct amd_comgr_metadata_node_s:
            unsigned long handle
        ```

        We can thus simplify the callback signature too

        ```cython
        void (*callback) (
          unsigned long key, unsigned long value, void * userdata)
        ```
    """
    result_list = ctypes.cast(
        ctypes.c_void_p(userdata), ctypes.POINTER(ctypes.py_object)
    ).contents.value
    key = _amd_comgr.amd_comgr_metadata_node_s(handle=ckey)
    result_list.append(metadata_string_get_bytes(key))


def metadata_map_get_keys(metadata_map: _amd_comgr.amd_comgr_metadata_node_s):
    """Get the keys of a AMD COMGR metadata map as `list`.

    Args:
        metadata_map (`~.amd_comgr_metadata_node_s`):
            A metadata node that represents a map.
    """
    keys = ctypes.py_object([])
    comgr_check(
        _amd_comgr.amd_comgr_iterate_map_metadata(
            metadata_map,
            ctypes.cast(_get_map_keys_cb, ctypes.c_void_p),
            ctypes.addressof(keys),
        )
    )
    return keys.value


def parse_metadata(metadata: _amd_comgr.amd_comgr_metadata_node_s, level: int = 0):
    """Parse metadata node and return a nest of `dict`, `list`, and `str`.

    Args:
        metadata_map (`~.amd_comgr_metadata_node_s`):
            A metadata node that represents a map.
        level (int, optional):
            Not used yet. Useful for debugging.
    """
    metadata_kind = comgr_check(_amd_comgr.amd_comgr_get_metadata_kind(metadata))
    if (
        metadata_kind
        == _amd_comgr.amd_comgr_metadata_kind_s.AMD_COMGR_METADATA_KIND_MAP
    ):
        result = {}
        map_keys = metadata_map_get_keys(metadata)
        for key in map_keys:
            map_value: _amd_comgr.amd_comgr_metadata_node_s = comgr_check(
                _amd_comgr.amd_comgr_metadata_lookup(metadata, key)
            )
            result[key.decode("utf-8")] = parse_metadata(map_value, level + 1)
            comgr_check(_amd_comgr.amd_comgr_destroy_metadata(map_value))
        return result
    elif (
        metadata_kind
        == _amd_comgr.amd_comgr_metadata_kind_s.AMD_COMGR_METADATA_KIND_LIST
    ):
        result = []
        list_size = comgr_check(
            _amd_comgr.amd_comgr_get_metadata_list_size(metadata)
        )
        assert isinstance(list_size, int)
        for i in range(0, list_size):
            list_entry: _amd_comgr.amd_comgr_metadata_node_s = comgr_check(
                _amd_comgr.amd_comgr_index_list_metadata(metadata, i)
            )
            result.append(parse_metadata(list_entry, level + 1))
            comgr_check(_amd_comgr.amd_comgr_destroy_metadata(list_entry))
        return result
    elif (
        metadata_kind
        == _amd_comgr.amd_comgr_metadata_kind_s.AMD_COMGR_METADATA_KIND_STRING
    ):
        return metadata_string_get_bytes(metadata).decode("utf-8")


def parse_data_metadata(data: _amd_comgr.amd_comgr_data_s):
    """Parse metadata of a `amd_comgr_data_s` object."""
    metadata = comgr_check(_amd_comgr.amd_comgr_get_data_metadata(data))
    result = parse_metadata(metadata)
    comgr_check(_amd_comgr.amd_comgr_destroy_metadata(metadata))
    return result


def parse_code_obj_metadata(
    code_obj,
    code_obj_size,
    kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE,
):
    """Parse metadata of a code object, e.g., one generated via HIPRTC.

    Args:
        code_obj:
            Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
            e.g. an implementor of the Python buffer protocol such as `bytes`.
        code_obj_size (`int`):
            Length of the code.
        kind (`~.amd_comgr_data_kind_s`, optional):
            Kind of the code object in terms of AMD COMGR kinds, e.g.
            `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
            which is the default.
    """
    data = comgr_check(_amd_comgr.amd_comgr_create_data(kind))
    comgr_check(_amd_comgr.amd_comgr_set_data(data, code_obj_size, code_obj))
    result = parse_data_metadata(data)
    comgr_check(_amd_comgr.amd_comgr_release_data(data))
    return result


def parse_code_obj_kernel_names(
    code,
    code_size,
    kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE,
):
    """Return the names of kernels in the code object.

    Results are returned in order of appearance.

    Args:
        code:
            Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
            e.g. an implementor of the Python buffer protocol such as `bytes`.
        code_size (`int`):
            Length of the code.
        kind (`~.amd_comgr_data_kind_s`, optional):
            Kind of the code object in terms of AMD COMGR kinds, e.g.
            `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
            which is the default.
    """
    metadata = parse_code_obj_metadata(code, code_size, kind)
    return [k[".name"] for k in metadata["amdhsa.kernels"]]


def get_isa_names(decode: bool = True) -> list:
    """Return list of ISA names supported by this COMGR version

    Return ISA names supported by this COMGR version as `list` of `str` or
    `bytes`.

    Args:
        decode (`bool`, optional):
            If the names should be decoded to a Python `str`.
    Returns:
        `list`:
            List of ISA names, either as Python `str` (``decode=True``) or
            `bytes`.
    """
    result = []
    num_isas = comgr_check(_amd_comgr.amd_comgr_get_isa_count())
    assert isinstance(num_isas, int)
    for i in range(0, num_isas):
        isa_name = comgr_check(_amd_comgr.amd_comgr_get_isa_name(i))
        result.append(
            isa_name.decode("utf-8") if decode else isa_name.encode("utf-8")
        )
    return result


def get_isa_metadata(isa_name):  # type: (str|bytes) -> ...
    """Parse metadata for a specific ISA.

    Args:
        isa_name (`bytes` or `str`):
            The ISA name
    See:
        get_isa_names
    """
    assert isinstance(isa_name, (bytes, str))
    if isinstance(isa_name, bytes):
        isa_name_bytes = isa_name
    else:
        isa_name_bytes = isa_name.encode("utf-8")
    isa_metadata: _amd_comgr.amd_comgr_metadata_node_s = comgr_check(
        _amd_comgr.amd_comgr_get_isa_metadata(isa_name_bytes)
    )
    result = parse_metadata(isa_metadata)
    comgr_check(_amd_comgr.amd_comgr_destroy_metadata(isa_metadata))
    return result


def get_isa_metadata_all():
    """Return metadata for all ISAs supported by this COMGR version as dict."""
    result = {}
    for isa_name in get_isa_names(decode=True):
        result[isa_name] = get_isa_metadata(isa_name)
    return result


class Symbol:
    """Represents a code symbol in a code object.

    Members result from `~.amd_comgr_symbol_get_info` supplied with the
    following ``attribute`` parameters:

    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME_LENGTH`:
        The length of the symbol name in bytes. Does not include the NUL
        terminator. The type of this attribute is uint64_t.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME`:
        The name of the symbol. The type of this attribute is character array
        with the length equal to the value of the
        AMD_COMGR_SYMBOL_INFO_NAME_LENGTH attribute plus 1 for a NUL
        terminator.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_TYPE`:
        The kind of the symbol. The type of this attribute is
        amd_comgr_symbol_type_t.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_SIZE`:
        Size of the variable. The value of this attribute is undefined if the
        symbol is not a variable. The type of this attribute is uint64_t.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED`:
        Indicates whether the symbol is undefined. The type of this attribute
        is bool.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_VALUE`:
        The value of the symbol. The type of this attribute is uint64_t.
    `~.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_LAST`:
        Marker for last valid symbol info.

    An object's member ``self.type`` can have the following values:

    ``UNKNOWN``:
        The symbol's type is unknown.
    ``NOTYPE``:
        The symbol's type is not specified.
    ``OBJECT``:
        The symbol is associated with a data object, such as a variable, an
        array, and so on.
    ``FUNC``:
        The symbol is associated with a function or other executable code.
    ``SECTION``:
        The symbol is associated with a section. Symbol table entries of this
        type exist primarily for relocation.
    ``FILE``:
        Conventionally, the symbol's name gives the name of the source file
        associated with the object file.
    ``COMMON``:
        The symbol labels an uninitialized common block.
    ``AMDGPU_HSA_KERNEL``:
        The symbol is associated with an AMDGPU Code Object V2 kernel function.
    """

    def __init__(self):
        self.type = None  # type: (str|None)
        self.name = None  # type: (str|None)
        self.size = -1  # type: (int)
        self.is_undefined = -1  # type: (int|bool)
        self.value = -1  # type: (int)


@ctypes.CFUNCTYPE(None, ctypes.c_ulong, ctypes.c_void_p)
def _iterate_symbols_cb(csymbol, userdata):
    """Callback for extracting keys from a metadata map.

    Intended to be used with amd_comgr_iterate_map_metadata(object metadata,
    object callback, object user_data)

    Note:
        The callback has the following signature:

        ```cython
        void (*callback) (amd_comgr_metadata_node_s key,
          amd_comgr_metadata_node_s value, void * userdata)
        ```

        where

        ```cython
        cdef struct amd_comgr_metadata_node_s:
            unsigned long handle
        ```

        We can thus simplify the callback signature too

        ```cython
        void (*callback) (unsigned long key, unsigned long value,
          void * userdata)
        ```
    """
    result_dict = ctypes.cast(
        ctypes.c_void_p(userdata), ctypes.POINTER(ctypes.py_object)
    ).contents.value
    result = Symbol()
    symbol = _amd_comgr.amd_comgr_symbol_s(handle=csymbol)

    # 1) type
    symbol_type = _amd_comgr.amd_comgr_symbol_type_s.ctypes_type()(0)
    _amd_comgr.amd_comgr_symbol_get_info(
        symbol,
        _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_TYPE,
        ctypes.addressof(symbol_type),
    )
    result.type = _amd_comgr.amd_comgr_symbol_type_s(
        symbol_type.value
    ).name.replace("AMD_COMGR_SYMBOL_TYPE_", "")
    # 2) name
    symbol_name_len = ctypes.c_uint64(0)
    _amd_comgr.amd_comgr_symbol_get_info(
        symbol,
        _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
        ctypes.addressof(symbol_name_len),
    )
    symbol_name = bytes(symbol_name_len.value)
    _amd_comgr.amd_comgr_symbol_get_info(
        symbol,
        _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_NAME,
        symbol_name,
    )
    result.name = symbol_name.decode("utf-8")
    #  3) size
    if result.type in ("OBJECT", "FUNC"):
        symbol_size = ctypes.c_uint64(0)
        _amd_comgr.amd_comgr_symbol_get_info(
            symbol,
            _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_SIZE,
            ctypes.addressof(symbol_size),
        )
        result.size = symbol_size.value
    # 4) is_undefined
    symbol_is_undefined = ctypes.c_bool(0)
    _amd_comgr.amd_comgr_symbol_get_info(
        symbol,
        _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED,
        ctypes.addressof(symbol_is_undefined),
    )
    result.is_undefined = symbol_is_undefined.value
    # 5) value
    symbol_value = ctypes.c_uint64(0)
    _amd_comgr.amd_comgr_symbol_get_info(
        symbol,
        _amd_comgr.amd_comgr_symbol_info_s.AMD_COMGR_SYMBOL_INFO_VALUE,
        ctypes.addressof(symbol_value),
    )
    result.value = symbol_value.value

    # add to result list
    result_dict[result.name] = result.__dict__


def parse_data_symbols(data: _amd_comgr.amd_comgr_data_s):
    """Parse all symbols of a data object and return as `dict`."""
    result = {}
    wrapper = ctypes.py_object(result)
    comgr_check(
        _amd_comgr.amd_comgr_iterate_symbols(
            data,
            ctypes.cast(_iterate_symbols_cb, ctypes.c_void_p),
            ctypes.addressof(wrapper),
        )
    )

    # Convert the virtual ELF addresses to the actual offset
    # in the code object.
    for symbol in result.values():
        if symbol["type"] in ("OBJECT", "FUNC"):
            code_object_offset = ctypes.c_uint64(
                0
            )  # TODO(interfacegen): Make return value
            slice_size = ctypes.c_uint64(
                0
            )  # TODO(interfacegen): Make return value
            nobits = ctypes.c_bool(
                False
            )  # TODO(interfacegen): Make return value
            comgr_check(
                _amd_comgr.amd_comgr_map_elf_virtual_address_to_code_object_offset(
                    data,
                    symbol["value"],
                    ctypes.addressof(code_object_offset),
                    ctypes.addressof(slice_size),
                    ctypes.addressof(nobits),
                )
            )
            symbol["nobits"] = nobits.value
            if nobits.value:
                symbol["code_object_offset"] = None
                symbol["slice_size"] = slice_size.value
            else:
                symbol["code_object_offset"] = code_object_offset.value
                symbol["slice_size"] = None

    return result


def parse_code_symbols(
    code,
    code_size,
    kind=_amd_comgr.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE,
):
    """Parse metadata of a code object, e.g., one generated via HIPRTC.

    Args:
        code:
            Code object that is accepted as input of `rocm.bindings.util.types.Pointer`,
            e.g. an implementor of the Python buffer protocol such as `bytes`.
        code_size (`int`):
            Length of the code.
        kind (`~.amd_comgr_data_kind_s`, optional):
            Kind of the code object in terms of AMD COMGR kinds, e.g.
            `~.amd_comgr_data_kind_s.AMD_COMGR_DATA_KIND_EXECUTABLE`,
            which is the default.
    """
    data = comgr_check(_amd_comgr.amd_comgr_create_data(kind))
    comgr_check(_amd_comgr.amd_comgr_set_data(data, code_size, code))
    result = parse_data_symbols(data)
    comgr_check(_amd_comgr.amd_comgr_release_data(data))
    return result


class Data:
    @staticmethod
    def kind_str_to_enum(kind_str: str):
        """Prepends ``AMD_COMGR_DATA_KIND_`` to ``kind_str`` and looks up enum.

        Note:
            Also converts ``kind_str`` to upper case.

        The following ``kind_str`` keys can be used (state: ROCm 6.0.0):

        UNDEF:
            No data is available.
        SOURCE:
            The data is a textual main source.
        INCLUDE:
            The data is a textual source that is included in the main source or
            other include source.
        PRECOMPILED_HEADER:
            The data is a precompiled-header source that is included in the
            main source or other include source.
        DIAGNOSTIC:
            The data is a diagnostic output.
        LOG:
            The data is a textual log output.
        BC:
            The data is compiler LLVM IR bit code for a specific isa.
        RELOCATABLE:
            The data is a relocatable machine code object for a specific isa.
        EXECUTABLE:
            The data is an executable machine code object for a specific isa.
            An executable is the kind of code object that can be loaded and
            executed.
        BYTES:
            The data is a block of bytes.
        FATBIN:
            The data is a fat binary (clang-offload-bundler output).
        AR:
            The data is an archive.
        BC_BUNDLE:
            The data is a bundled bitcode.
        AR_BUNDLE:
            The data is a bundled archive.
        LAST:
            Marker for last valid data kind.

        The following ``kind_str`` can be used with ROCm 6.2+:

        OBJ_BUNDLE:
            The data is an object file bundle.

        The following ``kind_str`` can be used with ROCm 6.4+:

        SPIRV:
            The data is SPIR-V IR.

        """
        return getattr(
            _amd_comgr.amd_comgr_data_kind_s,
            "AMD_COMGR_DATA_KIND_" + kind_str.upper(),
        )

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._data = None
        instance._name = None
        instance._len = None
        instance.kind_str = None
        instance.source_bytes = None
        return instance

    def __init__(self, name: str, kind_str: str, source_buffer=None):
        self._data = comgr_check(
            _amd_comgr.amd_comgr_create_data(Data.kind_str_to_enum(kind_str))
        )
        self.kind_str = kind_str
        if source_buffer:
            self._set_data_buffer(source_buffer)
        self._set_data_name(name)

    def _set_data_name(self, name):
        self._name = to_bytes(name).decode("utf-8")
        comgr_check(
            _amd_comgr.amd_comgr_set_data_name(
                self.get(), CStr(self._name)
            )
        )

    def _set_data_buffer(self, data_buffer=None):
        self.source_bytes = to_bytes(data_buffer)  # store to keep alive
        comgr_check(
            _amd_comgr.amd_comgr_set_data(
                self.get(), len(self.source_bytes), self.source_bytes
            )
        )

    def get_data_name(self) -> str:
        """Returns the data object's name as `str`."""
        name_len = ctypes.c_ulong(0)
        comgr_check(
            _amd_comgr.amd_comgr_get_data_name(
                self.get(), ctypes.addressof(name_len), None
            )
        )
        name_len.value -= 1  # strip NUL char
        buf = bytes(name_len.value)
        comgr_check(
            _amd_comgr.amd_comgr_get_data_name(
                self.get(), ctypes.addressof(name_len), buf
            )
        )
        return buf.decode("utf-8")

    def get_data_len(self):  # type(Data) -> int
        """Get the size of the managed data as Python 'int'."""
        if not self._len:
            data_len = ctypes.c_ulong(0)
            comgr_check(
                _amd_comgr.amd_comgr_get_data(
                    self.get(), ctypes.addressof(data_len), None
                )
            )
            self._len = data_len.value
        return self._len

    def get_data_bytes(self) -> bytes:
        """Copy the managed data into a new buffer and return it as Python
        'bytes'.

        Note:
            This routine should only be used if this is a result data object,
            e.g. obtained as result from an action. If this is a source data
            object, you can also access ``self.source_bytes`` for a copy of
            the original source data buffer.
        """
        data_len = ctypes.c_ulong(0)
        comgr_check(
            _amd_comgr.amd_comgr_get_data(
                self.get(), ctypes.addressof(data_len), None
            )
        )
        buf = bytes(data_len.value)
        comgr_check(
            _amd_comgr.amd_comgr_get_data(
                self.get(), ctypes.addressof(data_len), buf
            )
        )
        return buf

    def __len__(self):
        return self.get_data_len()

    def __del__(self):
        comgr_check(_amd_comgr.amd_comgr_release_data(self.get()))

    def get(self):
        return self._data


class DataSet:
    def __init__(self, *datas):
        self._data_set = comgr_check(_amd_comgr.amd_comgr_create_data_set())
        self.datas = []  # keep the objects alive
        for data in datas:
            self.add_data(data)

    def add_data(self, data: Data):
        self.datas.append(data)
        comgr_check(_amd_comgr.amd_comgr_data_set_add(self.get(), data.get()))

    def count_data(self, kind_str: str):
        return comgr_check(
            _amd_comgr.amd_comgr_action_data_count(
                self.get(), Data.kind_str_to_enum(kind_str)
            )
        )

    def get_data(self, kind_str: str, index: int) -> Data:
        data = Data.__new__(Data)
        data._data = comgr_check(
            _amd_comgr.amd_comgr_action_data_get_data(
                self.get(), Data.kind_str_to_enum(kind_str), index
            )
        )
        data.kind_str = kind_str
        return data

    def __del__(self):
        self.datas.clear()
        comgr_check(_amd_comgr.amd_comgr_destroy_data_set(self.get()))

    def get(self):
        return self._data_set


class Action:
    @staticmethod
    def action_kind_str_to_enum(action_kind_str: str):
        """Prepends ``AMD_COMGR_LANGUAGE_`` to ``action_kind_str`` and looks up
        enum.

        Note:
            Also converts ``action_kind_str`` to upper case.

        The following ``action_kind_str`` keys can be used (state: ROCm 6.0.0):

        SOURCE_TO_PREPROCESSOR:
            Preprocess each source data object in input in order. For each
            successful preprocessor invocation, add a source data object to
            result. Resolve any include source names using the names of
            includedata objects in input. Resolve any include relative path
            names using the working directory path in info. Preprocess the
            source for the language in info.
        ADD_PRECOMPILED_HEADERS:
            Copy all existing data objects in input to output, then add the
            device-specific and language-specific precompiled headers required
            for compilation.
        COMPILE_SOURCE_TO_BC:
            Compile each source data object in input in order. For each
            successful compilation add a bc data object to result. Resolve any
            include source names using the names of include data objects in
            input. Resolve any include relative path names using the working
            directory path in info. Produce bc for isa name in info. Compile
            the source for the language in info.
        ADD_DEVICE_LIBRARIES:
            (Removed in ROCm 6.4+) Copy all existing data objects in input to
            output, then add the device-specific and language-specific bitcode
            libraries required for compilation.
        LINK_BC_TO_BC:
            Link a collection of bitcodes, bundled bitcodes, and bundled
            bitcode archives in into a single composite (unbundled) bitcode.
            Any device library bc data object must be explicitly added to input
            if needed.
        OPTIMIZE_BC_TO_BC:
            Optimize each bc data object in input and create an optimized bc
            data object to result.
        CODEGEN_BC_TO_RELOCATABLE:
            Perform code generation for each bc data object in input in order.
            For each successful code generation add a relocatable data object
            to result.
        CODEGEN_BC_TO_ASSEMBLY:
            Perform code generation for each bc data object in input in order.
            For each successful code generation add an assembly source data
            object to result.
        LINK_RELOCATABLE_TO_RELOCATABLE:
            Link each relocatable data object in input together and add the
            linked relocatable data object to result. Any device library
            relocatable data object must be explicitly added to input if
            needed.
        LINK_RELOCATABLE_TO_EXECUTABLE:
            Link each relocatable data object in input together and add the
            linked executable data object to result. Any device library
            relocatable data object must be explicitly added to input if
            needed.
        ASSEMBLE_SOURCE_TO_RELOCATABLE:
            Assemble each source data object in input in order into machine
            code. For each successful assembly add a relocatable data object to
            result. Resolve any include source names using the names of include
            data objects in input. Resolve any include relative path names
            using the working directory path in info. Produce relocatable for
            isa name in info.
        DISASSEMBLE_RELOCATABLE_TO_SOURCE:
            (Deprecated from ROCm 7.1+ on)
            Disassemble each relocatable data object in input in order. For
            each successful disassembly add a source data object to result.
        DISASSEMBLE_EXECUTABLE_TO_SOURCE:
            (Deprecated from ROCm 7.1+ on) Disassemble each executable data
            object in input in order. For each successful disassembly add a
            source data object to result.
        DISASSEMBLE_BYTES_TO_SOURCE:
            (Deprecated from ROCm 7.1+ on) Disassemble each bytes data object
            in input in order. For each successful disassembly add a source
            data object to result. Only simple assembly language commands are
            generate that corresponf to raw bytes are supported, not any
            directives that control the code object layout, or symbolic branch
            targets or names.
        COMPILE_SOURCE_TO_FATBIN:
            Compile each source data object in input in order. For each
            successful compilation add a fat binary to result. Resolve any
            include source names using the names of include data objects in
            input. Resolve any include relative path names using the working
            directory path in info. Produce fat binary for isa name in info.
            Compile the source for the language in info.
        COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC:
            Compile each source data object in input in order. For each
            successful compilation add a bc data object to result. Resolve any
            include source names using the names of include data objects in
            input. Resolve any include relative path names using the working
            directory path in info. Produce bc for isa name in info. Compile
            the source for the language in info. Link against the
            device-specific and language-specific bitcode device libraries
            required for compilation.
        LAST:
            Marker for last valid action kind.

        The following ``action_kind_str`` keys can be used with ROCm 6.2+:

        UNBUNDLE:
            Unbundle each source data object in input. These objects can be
            bitcode bundles, or an archive containing bitcode bundles. For each
            successful unbundling, add a bc object or archive object to result,
            depending on the corresponding input.

        The following ``action_kind_str`` keys can be used with ROCm 6.4+:

        COMPILE_SOURCE_TO_RELOCATABLE:
            Compile a single source data object in input in order. For each
            successful compilation add a relocatable data object to result.
        COMPILE_SOURCE_TO_EXECUTABLE:
            Compile each source data object in input and create a single
            executable. For each successful compilation add a relocatable data
            object to result.
        TRANSLATE_SPIRV_TO_BC:
            Translate each source SPIR-V object in input into LLVM IR Bitcode.
            For each successful translation, add a bc object to p result.

        The following ``action_kind_str`` keys can be used with ROCm 6.4+:

        """
        return getattr(
            _amd_comgr.amd_comgr_action_kind_s,
            "AMD_COMGR_ACTION_" + action_kind_str.upper(),
        )

    @staticmethod
    def lang_str_to_enum(lang_str: str):
        """Prepends ``AMD_COMGR_LANGUAGE_`` to ``lang_str`` and looks up enum.

        Note:
            Also converts ``lang_str`` to upper case.

        The following ``lang_str`` keys can be used (state: ROCm 6.0.0):

        NONE:
            No high level language.
        OPENCL_1_2:
            OpenCL 1.2.
        OPENCL_2_0:
            OpenCL 2.0.
        HC:
            AMD Hetrogeneous C++ (HC).
        HIP:
            HIP.
        LAST:
            Marker for last valid language.
        """
        return getattr(
            _amd_comgr.amd_comgr_language_s,
            "AMD_COMGR_LANGUAGE_" + lang_str.upper(),
        )

    def __init__(
        self,
        action_kind_str: str,
        isa_name=None,
        lang_str=None,
        options=None,
        logging: bool = False,
    ):
        self._action_info = comgr_check(_amd_comgr.amd_comgr_create_action_info())
        self.result_data_set = DataSet()
        self._action_kind = Action.action_kind_str_to_enum(action_kind_str)
        if isa_name:
            self.set_isa_name(isa_name)
        if lang_str:
            self.set_language(lang_str)
        if options:
            self.set_options(options)
        self.set_logging(logging)

    def set_isa_name(self, isa_name):
        """Sets the action info object's isa_name.

        Args:
            isa_name (`str` or Python buffer such as `bytes`):
                ISA name supported by this version of AMD COMGR, e.g.
                ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
                See `~.get_isa_names`, `~.get_isa_metadata_all` for more
                information.
        Note:
            Input will be null-terminated if it is not already.
        """
        comgr_check(
            _amd_comgr.amd_comgr_action_info_set_isa_name(
                self._action_info,
                CStr(isa_name),
            )
        )

    def set_language(self, lang_str: str):
        """Set the language of the input data, e.g. "HIP".

        See:
            `lang_str_to_enum`.
        """
        comgr_check(
            _amd_comgr.amd_comgr_action_info_set_language(
                self.get(), Action.lang_str_to_enum(lang_str)
            )
        )

    def set_options(
        self, options  # type: (list|tuple)
    ):
        """Set options to supply to the action runner.

        Args:
            options (`list` or `tuple` of Python buffer such as `bytes`):
                Options to supply to the action runner.
        Note:
            Input will be null-terminated if it is not already.
        """
        # NOTE: Function `amd_comgr_action_info_set_option_list` wraps
        #       std::string around C string options inputs, which
        #       implies that a copy of each option input is created.
        #       Therefore, the inputs don't strictly need to outlive
        #       this call. We still wrap each option in CStr because
        #       (a) it gives a content-deduplicated, NUL-terminated
        #       buffer for free, and (b) the per-content intern
        #       remains bounded by the number of unique compile flags
        #       ever passed.
        comgr_check(
            _amd_comgr.amd_comgr_action_info_set_option_list(
                self.get(), [CStr(o) for o in options], len(options)
            )
        )

    def get_num_options(self):

        return comgr_check(
            _amd_comgr.amd_comgr_action_info_get_option_list_count(self.get())
        )

    def get_option(self, index: int) -> bytes:
        # First determines length of string, then loads it into buffer.
        str_len = ctypes.c_ulong(0)
        comgr_check(
            _amd_comgr.amd_comgr_action_info_get_option_list_item(
                self.get(), index, ctypes.addressof(str_len), None
            )
        )
        str_len.value -= 1  # remove the 0x00 terminator
        str_buf = bytes(str_len.value)
        comgr_check(
            _amd_comgr.amd_comgr_action_info_get_option_list_item(
                self.get(), index, ctypes.addressof(str_len), str_buf
            )
        )
        return str_buf

    def set_logging(self, logging: bool):
        """Enable logging."""
        comgr_check(
            _amd_comgr.amd_comgr_action_info_set_logging(self.get(), logging)
        )

    def do_action(
        self, input_data_set: DataSet, check=True
    ) -> _amd_comgr.amd_comgr_status_s:
        """Run the action for the given data_set."""
        result = _amd_comgr.amd_comgr_do_action(
            self._action_kind,
            self.get(),
            input_data_set.get(),
            self.result_data_set.get(),
        )

        if check:
            comgr_check(result)
        return result[0]  # note: always a tuple

    def get(self):
        return self._action_info

    def __del__(self):
        comgr_check(_amd_comgr.amd_comgr_destroy_action_info(self.get()))


class _DisassemblyUserData:
    """User object for storing input and output for the disassembly task."""

    @staticmethod
    def from_ctypes_c_void_p(
        ptr,
    ):  # type: (ctypes.c_void_p) -> _DisassemblyUserData
        return ctypes.cast(
            ctypes.c_void_p(ptr), ctypes.POINTER(ctypes.py_object)
        ).contents.value  # type: _DisassemblyUserData

    def __init__(self, program):  # type: (bytes) -> None
        self.program = program  # type: bytes
        self.disassembly = []  # type: list
        self.append_address_annotation = False  # type: bool


@ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
)
def _default_read_memory_cb(
    cursor, dest, size, user_data
):  # type: (ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p) -> ctypes.c_uint64  # noqa: E501
    """Callback: Copies the next program memory block into the buffer provided
    by COMGR, increments the read position.

    The user_data is assumed to be an object that implements `__len__(self)`.

    References:

    * https://github.com/rocm/llvm-project/blob/release/rocm-rel-7.0/amd/comgr/test/disasm_instr_test.c
    * https://github.com/ROCm/llvm-project/blob/release/rocm-rel-7.0/amd/comgr/include/amd_comgr.h.in

    Equivalent C signature

    uint64_t read_memory_cb(uint64_t From, char *To, uint64_t Size, void *UserData)

    Note:
        We map the ``char*`` to `ctypes.c_void_p` because c_char_p would be automatically converted to `bytes`.

    Note:
        Implicitly assumes that the passed program is of Python type `bytes`.
    """  # noqa: E501
    _disassembly_user_data = _DisassemblyUserData.from_ctypes_c_void_p(
        user_data
    )
    program = _disassembly_user_data.program

    if cursor >= len(program):
        return 0

    # tailor copy length based on size of buffer and program length
    if cursor + size > len(program):
        size = len(program) - cursor

    source_buffer = ctypes.create_string_buffer(
        program[cursor : cursor + size]
    )
    ctypes.memmove(dest, source_buffer, size)
    return size


# void (cnst char *Instruction, void *UserData)
@ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
def _default_append_instruction_cb(
    instruction, user_data
):  # type: (ctypes.c_char_p, ctypes.c_void_p) -> None
    """Append decoded instruction to list of already disassembled instructions.

    Equivalent C function signature:

    void print_instruction_cb(const char *instruction, void *user_data)
    """

    _disassembly_user_data = _DisassemblyUserData.from_ctypes_c_void_p(
        user_data
    )
    _disassembly_user_data.disassembly.append(
        instruction.decode().strip()
    )  # Convert 0-terminated c_char_p to str


# void (cnst char *Instruction, void *UserData)
@ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p)
def _default_append_address_annotation_cb(
    address, user_data
):  # type: (ctypes.c_uint64, ctypes.c_void_p) -> None
    """Append decoded address annotation to last processed instruction.

    Equivalent C function signature:

    void print_address_annotation_cb(uint64_t address, void *user_data),
    """

    _disassembly_user_data = _DisassemblyUserData.from_ctypes_c_void_p(
        user_data
    )
    if _disassembly_user_data.append_address_annotation:
        last_line = _disassembly_user_data.disassembly[-1]
        _disassembly_user_data.disassembly[-1] = (
            last_line + " ; addr: " + str(hex(address))
        )  # Convert 0-terminated c_char_p to str


def disassemble_program(
    isa_name,
    program,
    append_address_annotation=False,
    read_memory_cb=_default_read_memory_cb,
    append_instruction_cb=_default_append_instruction_cb,
    append_address_annotation_cb=_default_append_address_annotation_cb,
):  # type: (...) -> str
    """Disassemble an AMD GPU machine code program.

    Note:
        DISASSEMBLE_* Actions will soon be deprecated;
        see: https://github.com/<internal-amd-org>/llvm-project/pull/2677

    Args:
        program:
            A block of machine code that represents a sequence of instructions.
    """
    program = to_bytes(program)

    disassembly_info = comgr_check(
        _amd_comgr.amd_comgr_create_disassembly_info(
            CStr(isa_name),
            ctypes.cast(read_memory_cb, ctypes.c_void_p),
            ctypes.cast(append_instruction_cb, ctypes.c_void_p),
            ctypes.cast(append_address_annotation_cb, ctypes.c_void_p),
        )
    )

    wrapped = _DisassemblyUserData(program)
    wrapped.append_address_annotation = append_address_annotation
    user_data = ctypes.py_object(wrapped)

    status = _amd_comgr.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS
    address = 0
    size = ctypes.c_ulong(0)  # TODO(interfacegen): Make return value
    while (
        status == _amd_comgr.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS
        and address < len(program)
    ):
        status = _amd_comgr.amd_comgr_disassemble_instruction(
            disassembly_info,
            address,
            ctypes.addressof(user_data),
            ctypes.addressof(size),
        )[0]
        address += size.value

    comgr_check(_amd_comgr.amd_comgr_destroy_disassembly_info(disassembly_info))
    return "\n".join(wrapped.disassembly) + "\n"


def _get_code_symbol_bytes(
    code_obj,
    symbols,
    symbol_name,
):
    """
    Args:
        code_obj:
            An object that can be converted to bytes.
        symbols (`dict`):
            Dict of symbols.
        symbol_name (`str`):
            The name of the object/function to lookup or `None`.
            If `None` is specified, the first found object/function
            is chosen.
    """
    if symbol_name not in symbols:
        raise KeyError(
            f"no symbol named '{symbol_name}' stored in the code object"
        )
    symbol = symbols[symbol_name]
    if symbol["nobits"]:
        raise ValueError(f"symbol '{symbol_name}' is created at runtime")
    offset = symbol["code_object_offset"]
    size = symbol["size"]
    return code_obj[offset : offset + size]


def disassemble_code_obj_function(
    code_obj,
    isa_name,
    func_name=None,
    append_address_annotation=False,
):  # type: (...) -> str
    """Disassembles a kernel or device function stored in a code object.

    Identifies the size and location of the function symbol in the code object
    and disassembles it.

    Args:
        code_obj:
            An object that can be converted to bytes.
        func_name (`str` or `None`):
            The name of the function lookup or `None`.
            If `None` is specified, the first found function
            is disassembled.
    """
    symbols = parse_code_symbols(code_obj, len(code_obj))
    if not func_name:
        func_name = next(
            k for k, symbol in symbols.items() if symbol["type"] == "FUNC"
        )
    if func_name not in symbols:
        if not func_name:
            raise KeyError("no function found in code object")
        raise KeyError(
            f"no function named '{func_name}' stored in the code object"
        )
    program = _get_code_symbol_bytes(code_obj, symbols, func_name)
    result = disassemble_program(isa_name, program, append_address_annotation)
    return result


def _isa_name_to_amdgpu_arch(isa_name):
    return isa_name.split("--")[-1].split(":")[0]


def dump_metadata_yaml(metadata_dict):  # type: (dict) -> str
    """
    Note:
        AMD HSA kernel metadata does not
        have list of lists or dict of dicts.
        We see only dict of lists, dict of values
        and list of dicts.
    """

    def handle_dict_(_, thedict):  # type: (str|None, dict) -> str
        result = ""
        # list means linebreak after 'key:` + indent_delta
        for k, v in thedict.items():
            if isinstance(v, list):
                result += f"{k}:\n"
                child_result = handle_list_(None, v)
                result += textwrap.indent(child_result, " " * 2)
            elif isinstance(v, dict):
                assert False, "did not expect dict entry"
            else:
                result += f"{k}: {handle_value_(k, v)}"
        return result

    def handle_list_(_, thelist):  # type: (str, list) -> str
        result = ""
        for entry in thelist:
            child_result = handle_child_(None, entry)
            for i, l in enumerate(child_result.splitlines(keepends=True)):
                if i == 0:
                    result += "- " + l
                else:
                    result += " " * 2 + l
        return result

    def handle_value_(name, thevalue):  # type: (str|None, str) -> str
        r"""Handle int/bool/string value.

        Typical numeric values are integers.
        However, the following YAML entries are booleans:

        -  '.uses_dynamic_stack'

        All other entries are strings.
        """
        if name and name.startswith(".uses_"):
            thevalue = "false" if int(thevalue) == 0 else "true"
        return str(thevalue) + "\n"

    def handle_child_(name, child):  # type: (str|None, str)-> str
        if isinstance(child, dict):
            return handle_dict_(name, child)
        elif isinstance(child, (list, tuple)):
            return handle_list_(name, child)
        else:
            return handle_value_(name, child)

    return handle_dict_(None, metadata_dict)


def disassemble_amdhsa_code_obj_v6_kernel(
    code_obj,
    isa_name,
    kernel_name=None,
    append_address_annotation=False,
    raw=False,
):  # type: (...) -> str
    """Disassembles a kernel stored in an AMD GPU code object v6.

    Identifies the size and location of the function symbol in the code object
    and disassembles it.

    Args:
        code_obj:
            An object that can be converted to bytes.
        func_name (`str` or `None`, ooptional):
            The name of the function lookup or `None`.
            If `None` is specified, the first found kernel
            is disassembled.
        raw (`bool`, optional):
            Just return raw instructions, do not prepend and
            append code object v6 specific ELF directions
            and metadata. Defaults to `False`.

    """
    metadata_dict = parse_code_obj_metadata(
        code_obj=code_obj, code_obj_size=len(code_obj)
    )
    kernel_names = [k[".name"] for k in metadata_dict["amdhsa.kernels"]]
    if kernel_name is None:
        kernel_name = kernel_names[0]
    elif kernel_name not in kernel_names:
        raise RuntimeError(
            f"no AMD GPU kernel '{kernel_name}' declared in code object's "
            "metadata"
        )

    instructions = disassemble_code_obj_function(
        code_obj=code_obj,
        isa_name=isa_name,
        func_name=kernel_name,
        append_address_annotation=append_address_annotation,
    )

    amdgpu_arch = _isa_name_to_amdgpu_arch(isa_name)

    if raw:
        return instructions
    else:
        result = textwrap.indent(
            textwrap.dedent(
                f"""\
            .amdgcn_target "{str(isa_name)}"
            .amdhsa_code_object_version 6
            .text
            .protected {str(kernel_name)}
            .globl {str(kernel_name)}
            .p2align 8
            .type {str(kernel_name)},@function
            """
            ),
            "\t",
        )

        symbols = parse_code_symbols(code_obj, len(code_obj))
        kd_symbol = _get_code_symbol_bytes(
            code_obj, symbols, kernel_name + ".kd"
        )

        kd_parse_result = (
            amd_hsa_kernel_descriptor.parse_amdgpu_code_obj_kernel_descriptor(
                kd_symbol, amdgpu_arch
            )
        )

        result += str(kernel_name) + ":\n"
        result += textwrap.indent(instructions, "\t")
        result += textwrap.indent(
            kd_parse_result.render_amdhsa_kernel_directive(
                kernel_name
            ).rstrip(),
            "\t",
        )

        # append reduced metdata yaml
        metadata_dict_reduced = dict(metadata_dict)
        metadata_dict_reduced["amdhsa.kernels"] = [
            k
            for k in metadata_dict["amdhsa.kernels"]
            if k[".name"] == kernel_name
        ]
        result += textwrap.dedent(
            """
            \t.amdgpu_metadata
            ---
            {body}...
            \t.end_amdgpu_metadata
            """
        ).format(body=dump_metadata_yaml(metadata_dict_reduced))

        return result


def _do_single_siso_action(
    action_kind,  # type: str
    data,  # type: Data
    isa_name,  # type: str|bytes
    lang_str,  # type: str
    output_kind,  # type: str
    options,  # type: list[str,bytes]
    logging=False,  # type: bool
    check=True,  # type: bool
):  # type: (...) -> tuple[_amd_comgr.amd_comgr_status_s, bytes, str|None, str|None] # noqa: E501
    r"""Performs a single single-input-single-output (SISO) action.

    Returns:
        `tuple`:
            A tuple of size 4 with the following components (in that order):

            1. `~.amd_comgr_status_s`:
               Enum constant indicating success or kind of error.
            2. `bytes` or `None`: The result.
            3. `str` or `None`: The log output.
            4. `str` or `None`: Diagnostics output if this was enabled
                via an option.
    """
    isa_name = to_str(isa_name)
    if not isa_name.startswith(
        "amdgcn-amd-amdhsa--gfx"
    ) and not isa_name.startswith("gfx"):
        raise ValueError(
            "Argument 'isa_name' must start with 'gfx' or "
            f"'amdgcn-amd-amdhsa--'; is: {isa_name}"
        )

    action = Action(
        action_kind_str=action_kind,
        isa_name=isa_name,
        lang_str=lang_str,
        logging=logging,
        options=options,
    )
    status = action.do_action(DataSet(data), check=False)
    success = status == _amd_comgr.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS
    if success:
        assert action.result_data_set.count_data(output_kind), output_kind
        result = action.result_data_set.get_data(
            output_kind, 0
        ).get_data_bytes()
    else:
        result = None
    if success and action.result_data_set.count_data("DIAGNOSTIC"):
        diagnostic = (
            action.result_data_set.get_data("DIAGNOSTIC", 0)
            .get_data_bytes()
            .decode()
        )
    else:
        diagnostic = None
    if logging:
        log = (
            action.result_data_set.get_data("LOG", 0).get_data_bytes().decode()
        )
    else:
        log = None
    if check and not success:
        if log:
            raise RuntimeError(
                f"action 'AMD_COMGR_ACTION_{action_kind}' "
                + "failed.\nLog output:\n----\n"
                + log
                + "\n----\n"
            )
        else:
            raise RuntimeError(
                f"action 'AMD_COMGR_ACTION_{action_kind}' "
                + "failed (logging disabled, no log available)"
            )
    return (status, result, log, diagnostic)


def _do_single_compile_action(
    source,  # type: str|bytes
    source_lang,  # type: str
    source_kind,  # type: str
    output_kind,  # type: str
    isa_name,  # type: str|bytes
    hip_version_tuple,  # type: tuple[int,int,int]
    action_kind,  # type: str
    default_opts,  # type: list[str|bytes]
    extra_opts=[],  # type: list[str|bytes]
    prepend_hiprtc_runtime_header=False,  # type: bool
    logging=False,  # type: bool
    check=True,  # type: bool
):  # type: (...) -> tuple[_amd_comgr.amd_comgr_status_s, bytes, str|None, str|None] # noqa: E501
    """Run compilation-specific single-input-single-output (SISO) action.

    Returns:
        `tuple`:
            A tuple of size 4 with the following components (in that order):

            1. `~.amd_comgr_status_s`:
                Enum constant indicating success or kind of error.
            2. `bytes` or `None`: The result.
            3. `str` or `None`: The log output.
            4. `str` or `None`: Diagnostics output if this was enabled
                via an option.
    """
    isa_name = to_str(isa_name)
    if not isa_name.startswith(
        "amdgcn-amd-amdhsa--gfx"
    ) and not isa_name.startswith("gfx"):
        raise ValueError(
            "Argument 'isa_name' must start with 'gfx' or "
            f"'amdgcn-amd-amdhsa--'; is: {isa_name}"
        )

    assert source_lang in ("HIP", "HSA", "BC", "LL")
    source_name = "source." + source_lang.lower()
    assert not prepend_hiprtc_runtime_header or source_lang == "HIP"

    if prepend_hiprtc_runtime_header and isinstance(source, str):
        source = HIPRTC_RUNTIME_HEADER + "\n" + source
    elif prepend_hiprtc_runtime_header:
        source = HIPRTC_RUNTIME_HEADER.encode() + b"\n" + bytes(source)

    # prepare options
    options = []
    if source_lang == "HIP":
        # trim expr like: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
        offload_arch = (
            to_bytes(isa_name).decode().replace("amdgcn-amd-amdhsa--", "")
        )
        options = [
            f"--offload-arch={offload_arch}",
            "--hip-version=" + ".".join(map(str, hip_version_tuple)),
            f"-DHIP_VERSION_MAJOR={hip_version_tuple[0]}",
            f"-DHIP_VERSION_MINOR={hip_version_tuple[1]}",
            f"-DHIP_VERSION_PATCH={hip_version_tuple[2]}",
        ]

        if prepend_hiprtc_runtime_header:
            default_opts.append("-D__HIPCC_RTC__")

    options += default_opts + extra_opts

    source_lang = "HIP"
    if source_lang == "HSA":
        source_lang = "HIP"
    elif not source_lang == "HIP":
        assert source_lang.startswith("BC")
        source_lang = "NONE"

    return _do_single_siso_action(
        action_kind,
        Data(source_name, source_kind, source),
        isa_name,
        source_lang,
        output_kind,
        options,
        logging,
        check,
    )


def compile_hip_to_bc(
    source,  # type: bytes|str
    isa_name,  # type: bytes|str
    hip_version_tuple,  # type: tuple[int,int,int]
    extra_opts=[],  # type: list[str|bytes]
    default_opts=[
        "-fgpu-rdc",
        "-O3",
        "-mcumode",
        "-std=c++14",
        "-nogpuinc",
        "-Wno-gnu-line-marker",
        "-Wno-missing-prototypes",
    ],  # type: list[str|bytes]
    prepend_hiprtc_runtime_header=False,  # type: bool
    logging=False,  # type: bool
    action_kind="COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC",  # type: str
):  # type: (...) -> tuple[bytes, str|None, str|None]
    """Compiles a HIP C++ source to LLVM BC.

    Args:
        source (`str` or Python buffer such as `bytes`):
            The input as bytes or str.
        isa_name (`str` or Python buffer such as `bytes`):
            ISA name supported by this version of AMD COMGR, e.g.
            ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
            See `~.get_isa_names`, `~.get_isa_metadata_all` for more
            information.
        hip_version_tuple (`tuple[int]`):
            Integer triple like ``(6,0,32830)`` that indicates a HIP version.
        extra_opts (`list` of `str` or Python buffer such as `bytes`):
            Extra options that are appended to the default options; see
            argument ``default_opts``.
            You would typically supply additional options via this value but
            can also use it overrule some or all of the options specified
            in default_opts. Defaults to `[]`.
        default_opts (`list` of `str` or Python buffer such as `bytes`):
            Default options that are typically not changed.
            Defaults to `["-fgpu-rdc", "-O3", "-mcumode", "-std=c++14",
            "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
        prepend_hiprtc_runtime_header (`bool`, optional):
            Prepend the hiprtc runtime header to the source code.
            Defaults to `False`.
        logging (`bool`):
            Enable logging. Defaults to ``False``.
        action_kind (`str`):
            The compile action kind. Defaults to
            ``"COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC"``
            Other supported option is ``"COMPILE_SOURCE_TO_BC"``.

    Returns:
        `tuple`:
            A `tuple` of size 1 with the following components (in that order):
            1. `bytes`: The compilation result, an LLVM BC file.
            2. `str` or `None`: The log output if logging was specified.
            3. `str` or `None`: The diagnostics output if diagnostics were
                enabled via options.

    Raises:
        `RuntimeError`:
            If one of the compile fails. Enable logging to get more
            detailed error reports.

    Note:
        String arguments are always encoded as `utf-8`.
    Note:
        Default option `-fgpu-rdc` keeps `__device__` functions in the bitcode
        file.
    See:
        `~.get_isa_names`, `~.get_isa_metadata_all`

    Note:
        This implementation is based on what AMD COMGR logs to screen when
        compiling HIP code to BC via hipRTC while the environment variables
        ``AMD_COMGR_REDIRECT_LOGS="stderr"`` and
        ``AMD_COMGR_EMIT_VERBOSE_LOGS=1`` are active:

        ```text
        ActionKind: AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC
        IsaName: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
        Options: "-O3" "-mcumode" "--hip-version=6.0.32830"
        "-DHIP_VERSION_MAJOR=6" "-DHIP_VERSION_MINOR=0"
        "-DHIP_VERSION_PATCH=32830" "-D__HIPCC_RTC__" "-include"
        "hiprtc_runtime.h" "-std=c++14" "-nogpuinc" "-Wno-gnu-line-marker"
        "-Wno-missing-prototypes" "--offload-arch=gfx90a:sramecc+:xnack-"
        "-fgpu-rdc"
        Path:
            Language: AMD_COMGR_LANGUAGE_HIP
        Compilation Args: [...]
        Driver Job Args: [...]
            ReturnStatus: AMD_COMGR_STATUS_SUCCESS
        ```
    """
    if action_kind not in (
        "COMPILE_SOURCE_TO_BC",
        "COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC",
    ):
        raise ValueError(
            "Argument 'action_kind' must be either 'COMPILE_SOURCE_TO_BC' "
            f"or 'COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC'; is: {action_kind}"
        )

    (_, result, log, diagnostic) = _do_single_compile_action(
        source,
        "HIP",
        "SOURCE",
        "BC",
        isa_name,
        hip_version_tuple,
        action_kind,
        default_opts,
        extra_opts,
        prepend_hiprtc_runtime_header,
        logging,
    )
    return (result, log, diagnostic)


def compile_bc_to_hsa(
    source,  # type: str|bytes
    isa_name,  # type: str|bytes
    bc_kind="BC",  # type: str|bytes
    extra_opts=[],  # type: list[str|bytes]
    logging=False,  # type: bool
):  # type: (...) -> tuple[bytes, str|None, str|None]
    """Translate LLVM IR/BC to HSA.

    Args:
        source (`str` or Python buffer such as `bytes`):
            The input as bytes or str.
        isa_name (`str` or Python buffer such as `bytes`):
            ISA name supported by this version of AMD COMGR, e.g.
            ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
            See `~.get_isa_names`, `~.get_isa_metadata_all` for more
            information.
        bc_kind (`str`, optional):
            Either "BC" or "BC_BUNDLE". Defaults to "BC".
        extra_opts (`list` of `str` or `bytes`-like, optional):
            Extra options that are appended to the default options.
            You would typically supply additional options via this value but
            can also use it overrule some or all of the options specified
            in default_opts. Defaults to `[]`.
        logging (`bool`, optional):
            Enable logging. Defaults to ``False``.

    Returns:
        `tuple`:
            A `tuple` of size 1 with the following components (in that order):

            1. `bytes`: The compilation result, an AMD GPU HSA assembly source
                file.
            2. `str` or `None`: The log output if logging was specified.
            3. `str` or `None`: The diagnostics output if diagnostics were
                enabled via options.

    Raises:
        `RuntimeError`:
            If one of the compile fails. Enable logging to get more
            detailed error reports.

    Note:
        String arguments are always encoded as `utf-8`.
    See:
        `~.get_isa_names`, `~.get_isa_metadata_all`
        ```
    """
    if bc_kind not in ("BC", "BC_BUNDLE"):
        raise ValueError(
            "Argument 'bc_kind' must be either 'BC' or 'BC_BUNDLE'"
        )

    (_, result, log, diagnostic) = _do_single_compile_action(
        source,
        "BC",  # source_lang
        bc_kind,
        "SOURCE",  # output_data_kind
        isa_name,
        None,
        "CODEGEN_BC_TO_ASSEMBLY",  # action_kind
        [],
        extra_opts,
        prepend_hiprtc_runtime_header=False,
        logging=logging,
    )

    return (result, log, diagnostic)


def compile_hip_to_hsa(
    source,  # type: str|bytes
    isa_name,  # type: str|bytes
    hip_version_tuple,  # type: tuple[int,int,int]
    extra_opts=[],  # type: list[str|bytes]
    default_opts=[
        "-S",
        "-O3",
        "-mcumode",
        "-std=c++14",
        "-nogpuinc",
        "-Wno-gnu-line-marker",
        "-Wno-missing-prototypes",
    ],  # type: list[str|bytes]
    prepend_hiprtc_runtime_header=False,  # type: bool
    logging=False,  # type: bool
):  # type: (...) -> tuple[bytes, str|None, str|None]
    """Compiles a HIP C++ source to HSA.

    Args:
        source (`str` or Python buffer such as `bytes`):
            The input as bytes or str.
        isa_name (`str` or Python buffer such as `bytes`):
            ISA name supported by this version of AMD COMGR, e.g.
            ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
            See `~.get_isa_names`, `~.get_isa_metadata_all` for more
            information.
        hip_version_tuple (`tuple[int]`):
            Integer triple like ``(6,0,32830)`` that indicates a HIP version.
        extra_opts (`list` of `str` or Python buffer such as `bytes`):
            Extra options that are appended to the default options; see
            argument ``default_opts``.
            You would typically supply additional options via this value but
            can also use it overrule some or all of the options specified
            in default_opts. Defaults to `[]`.
        default_opts (`list` of `str` or Python buffer such as `bytes`):
            Default options that are typically not changed.
            Defaults to `["-S", "-O3", "-mcumode", "-std=c++14",
            "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
        prepend_hiprtc_runtime_header (`bool`, optional)
            Prepend the hiprtc runtime header to the source code.
            Defaults to `False`.
        logging (bool):
            Enable logging. Defaults to ``False``.

    Returns:
        `tuple`:
            A `tuple` of size 3 with the following components (in that order):
            1. `bytes`: The compilation result, an AMD GPU HSA assembly source
               file.
            2. `str` or `None`: The log output if logging was specified.
            3. `str` or `None`: The diagnostics output if diagnostics were
                enabled via options.

    Raises:
        `RuntimeError`:
            If one of the compile fails. Enable logging to get more
            detailed error reports.

    Note:
        String arguments are always encoded as `utf-8`.
    See:
        `~.get_isa_names`, `~.get_isa_metadata_all`
        ```
    """

    (_, result, log, diagnostic) = _do_single_compile_action(
        source,
        "HIP",
        "SOURCE",
        "RELOCATABLE",
        isa_name,
        hip_version_tuple,
        "COMPILE_SOURCE_TO_RELOCATABLE",
        default_opts,
        extra_opts,
        prepend_hiprtc_runtime_header,
        logging,
    )

    return (result, log, diagnostic)


def compile_bc(
    ir_or_bc,  # type: str|bytes
    isa_name,  # type: str|bytes
    bc_kind="BC",  # type: str
    extra_opts=[],  # type: list[str|bytes]
    default_opts=[
        "-O3",
    ],  # type: list[str|bytes]
    logging=False,  # type: bool
):  # type: (...) -> tuple[bytes, str|None, str|None]
    """Compile LLVM BC/IR to AMD GPU code object.

    Args:
        ir_or_bc (`str` or Python buffer such as `bytes`):
            The input as bytes or str.
        isa_name (`str` or Python buffer such as `bytes`):
            ISA name supported by this version of AMD COMGR, e.g.
            ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
            See `~.get_isa_names`, `~.get_isa_metadata_all` for more
            information.
        bc_kind (`str`, optional):
            Either "BC" or "BC_BUNDLE". Defaults to "BC".
        extra_opts (`list` of `str` or Python buffer such as `bytes`):
            Extra options that are appended to the default options; see
            argument ``default_opts``.
            You would typically supply additional options via this value but
            can also use it overrule some or all of the options specified
            in default_opts. Defaults to `[]`.
        default_opts (`list` of `str` or Python buffer such as `bytes`):
            Default options that are typically not changed.
            Defaults to `["-S", "-O3", "-mcumode", "-std=c++14",
            "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes"]`.
        logging (bool):
            Enable logging. Defaults to ``False``.

    Returns:
        `tuple`:
            A `tuple` of size 3 with the following components (in that order):
            1. `bytes`: The compilation result, an AMD GPU object in ELF
               format.
            2. `str` or `None`: The log output if logging was specified.
            3. `str` or `None`: The diagnostics output if diagnostics were
               enabled via options.

    Raises:
        `RuntimeError`:
            If one of the compile fails. Enable logging to get more
            detailed error reports.

    Note:
        We deduced the order of actions and their parametrization from the
        AMD COMGR log output when running hiprtcLinkComplete with
        BC input. We decided to skip the LINK_BC_TO_BC step.
    """
    # We found that the following option is not needed for
    # single bc input.
    # (_, result, _, _) = _do_single_siso_action(
    #     "LINK_BC_TO_BC",
    #     Data("input.bc", "BC", ir_or_bc),
    #     isa_name,
    #     "HIP",
    #     output_kind="BC",
    #     options=[],
    #     logging=logging,
    # )
    result = ir_or_bc

    num_actions = 2
    logs, diagnostics = [""] * num_actions, [""] * num_actions
    (_, result, logs[0], diagnostics[0]) = _do_single_siso_action(
        "CODEGEN_BC_TO_RELOCATABLE",
        Data("input.bc", bc_kind, result),
        isa_name,
        "NONE",
        output_kind="RELOCATABLE",
        options=default_opts + extra_opts,
        logging=logging,
    )

    (_, result, logs[1], diagnostics[1]) = _do_single_siso_action(
        "LINK_RELOCATABLE_TO_EXECUTABLE",
        Data("linked.o", "RELOCATABLE", result),
        isa_name,
        "NONE",
        output_kind="EXECUTABLE",
        options=[],
        logging=logging,
    )

    if logging:
        log = "\n---".join(e for e in logs if e)
    else:
        log = None

    diagnostic = "\n---".join(e for e in diagnostics if e)
    if not len(diagnostic):
        diagnostic = None

    return result, log, diagnostic


def compile_hsa(
    hsa,  # type: str|bytes
    isa_name,  # type: str|bytes
    extra_opts=[],  # type: list[str|bytes]
    default_opts=[],  # type: list[str|bytes]
    logging=False,  # type: bool
):  # type: (...) -> tuple[bytes, str|None, str|None]
    """Compile AMD HSA assembly to AMD GPU code object.

    Args:
        source (`str` or Python buffer such as `bytes`):
            The input as bytes or str.
        isa_name (`str` or Python buffer such as `bytes`):
            ISA name supported by this version of AMD COMGR, e.g.
            ``amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-``.
            See `~.get_isa_names`, `~.get_isa_metadata_all` for more
            information.
        extra_opts (`list` of `str` or Python buffer such as `bytes`):
            Extra options that are appended to the default options; see
            argument ``default_opts``.
            You would typically supply additional options via this value but
            can also use it overrule some or all of the options specified
            in default_opts. Defaults to `[]`.
        default_opts (`list` of `str` or Python buffer such as `bytes`):
            Default options that are typically not changed.
            Defaults to `[]`.
        logging (bool):
            Enable logging. Defaults to ``False``.

    Returns:
        `tuple`:
            A `tuple` of size 3 with the following components (in that order):
            1. `bytes`: The compilation result, an AMD GPU object in ELF
               format.
            2. `str` or `None`: The log output if logging was specified.
            3. `str` or `None`: The diagnostics output if diagnostics were
               enabled via options.

    Raises:
        `RuntimeError`:
            If one of the compile fails. Enable logging to get more
            detailed error reports.

    Note:
        We deduced the order of actions and their parametrization from the
        AMD COMGR log output when running hiprtcLinkComplete with
        BC input. We decided to skip the LINK_BC_TO_BC step.
    """
    result = hsa

    num_actions = 2
    logs, diagnostics = [""] * num_actions, [""] * num_actions
    (_, result, logs[0], diagnostics[0]) = _do_single_siso_action(
        "ASSEMBLE_SOURCE_TO_RELOCATABLE",
        Data("input.s", "SOURCE", result),
        isa_name,
        "HIP",
        output_kind="RELOCATABLE",
        options=default_opts + extra_opts,
        logging=logging,
    )
    (_, result, logs[1], diagnostics[1]) = _do_single_siso_action(
        "LINK_RELOCATABLE_TO_EXECUTABLE",
        Data("linked.o", "RELOCATABLE", result),
        isa_name,
        "NONE",
        output_kind="EXECUTABLE",
        options=[],
        logging=logging,
    )

    if logging:
        log = "\n---".join(e for e in logs if e)
    else:
        log = None

    diagnostic = "\n---".join(e for e in diagnostics if e)
    if not len(diagnostic):
        diagnostic = None

    return result, log, diagnostic


def disassemble_via_action_deprecated(
    code_obj,  # type: bytes
    isa_name,  # type: str|bytes
    logging=False,  # type: bool
    action_kind="DISASSEMBLE_EXECUTABLE_TO_SOURCE",  # type: str
):  # type: (...) -> tuple[bytes, str, str]
    """Disassemble an AMD GPU executable/relocatable.

    Warning:
        DISASSEMBLE_* Actions will soon be deprecated;
        see: https://github.com/<internal-amd-org>/llvm-project/pull/2677

    """
    if action_kind not in (
        "DISASSEMBLE_EXECUTABLE_TO_SOURCE",
        "DISASSEMBLE_RELOCATABLE_TO_SOURCE",
    ):
        raise ValueError(f"action kind '{str(action_kind)}' not supported")

    data_set = DataSet()
    for sk in ("EXECUTABLE", "RELOCATABLE"):
        if sk in action_kind:
            input_data = Data("gpu-code-obj", sk, code_obj)
            data_set.add_data(input_data)

    action = Action(
        action_kind_str=action_kind,
        isa_name=isa_name,
        lang_str="HIP",
        logging=logging,
    )
    action.do_action(data_set)
    result = action.result_data_set.get_data("SOURCE", 0).get_data_bytes()
    if logging:
        log = (
            action.result_data_set.get_data("LOG", 0).get_data_bytes().decode()
        )
    else:
        log = None
    if action.result_data_set.count_data("DIAGNOSTIC"):
        diagnostic = (
            action.result_data_set.get_data("LOG", 0).get_data_bytes().decode()
        )
    else:
        diagnostic = None
    return (result, log, diagnostic)


with open(
    os.path.join(os.path.dirname(__file__), "hiprtc_runtime.h"), "r"
) as infile:
    HIPRTC_RUNTIME_HEADER = infile.read()
