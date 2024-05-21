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

"""Cache for link-time dependencies.

This module contains a simple cache that records arbitrary objects for triples of 
file content, compiler options, and target architecture. 
The cache is intended for caching linker dependencies for Numba HIP.

The cache does compute a cache key from file names but always the file content
so that 

This module delegates all calls that cannot be resolved via ``__getattribute__``
to the `LinkerCache` singleton.
"""

import hashlib


class LinkerCache:

    __INSTANCE = None

    @classmethod
    def get(cls):
        if not cls.__INSTANCE:
            cls.__INSTANCE = LinkerCache()
        return cls.__INSTANCE

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _make_cache_key(buffer, arch, opts, clean_str_key_components: bool = True):
        """Make a key from the input triple '`buffer`'-``arch``-``opts``.

        Args:
            clean_str_key_components (`bool`):
                Remove whitespace from components of type `str` in order to
                create the same key from strings like `-a -b`, `-a-b` and `-a      -b`.

        Note:
            All key constituents must be hashable. Strings must be encoded in "utf-8" format.
        """
        m = hashlib.md5()
        for key_component in (buffer, arch, opts):
            if isinstance(key_component, str):
                if clean_str_key_components:
                    key_component = key_component.translate(
                        str.maketrans("", "", " \t\n\r")
                    )
                key_component = key_component.encode("utf-8")
            m.update(key_component)
        return m.digest()

    def get_or_insert_entry_for_buffer(
        self,
        buffer,
        arch,
        opts="",
        entry=None,
        clean_str_key_components: bool = True,
    ):
        """Retrieves (entry == None) or inserts (entry != None) an entry for the given (``buffer``, ``arch``, ``opts``) triple.

        Note:
            Returns the entry also in the insertion case.

        Note:
            Despite the very similar name, this routine does not behave like
            llvmlite's `get_or_insert_<llvm_entity>` routines.

        Arguments:
            buffer (readable buffer or `str`):
                Content of a file in bytes.
            arch (readable buffer or `str`):
                A string identifying the architecture.
            opts (readable buffer or `str`):
                A string of options.
            entry (`object`):
                The entry to store.
            clean_str_key_components (`bool`):
                Remove whitespace chars in arguments that are of `str` type(!) before creating the key.
                Defaults to ``True``.

        Raises:
            `KeyError`: If argument ``entry`` is ``None`` and no entry is specified.
        """
        key = self._make_cache_key(buffer, arch, opts, clean_str_key_components)
        if entry == None:
            return self._cache[key]  # may fail with key error
        else:
            self._cache[key] = entry
            return entry

    def get_or_insert_entry_for_file(self, filepath: str, *args, **kwargs):
        """Variant of get_or_insert_entry_for_buffer that takes a file path instead of a buffer."""
        with open(filepath, "r") as infile:
            return self.get_or_insert_entry_for_buffer(infile.read(), *args, **kwargs)

    def delete_entry_for_buffer(
        self, buffer, arch, opts, clean_str_key_components: bool = True
    ):
        del self._cache[
            self._make_cache_key(buffer, arch, opts, clean_str_key_components)
        ]

    def delete_entry_for_file(self, filepath: str, *args, **kwargs):
        with open(filepath, "r") as infile:
            return self.delete_entry_for_buffer(infile.read(), *args, **kwargs)

    def clear(self):
        self._cache.clear()


_cache = LinkerCache.get()._cache
_make_cache_key = LinkerCache.get()._make_cache_key
get_or_insert_entry_for_buffer = LinkerCache.get().get_or_insert_entry_for_buffer
get_or_insert_entry_for_file = LinkerCache.get().get_or_insert_entry_for_file
delete_entry_for_buffer = LinkerCache.get().delete_entry_for_buffer
delete_entry_for_file = LinkerCache.get().delete_entry_for_file
clear = LinkerCache.get().clear
