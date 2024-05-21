#!/usr/bin/env -S python3 -m pytest -v -s
# MIT License
#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

import textwrap
import pprint
import tempfile
import os

from numba.hip.util import linkercache

ARCHS = ["gfx90a", "gfx942"]

BUFS = [b"bytes\nbytes\n", "str\nstr"]

OPTS = [
    "-fgpu-rdc -O3",
    "-fgpu-rdc        -O3",
    "     -fgpu-rdc        -O3",
]

ENTRIES = [
    "ENTRY_1",
    "ENTRY_2",
]


def test_00_make_cache_key():
    # Allow whitespace differences
    assert linkercache._make_cache_key(
        buffer=BUFS[0] + b" \t ",
        arch=ARCHS[0],
        opts="-fgpu-rdc -O3",
        clean_str_key_components=True,
    ) != linkercache._make_cache_key(
        buffer=BUFS[0],
        arch=ARCHS[0] + "     \t",
        opts="    -fgpu-rdc -O3  ",
        clean_str_key_components=True,
    )
    # don't allow whitespace differences
    assert linkercache._make_cache_key(
        buffer=BUFS[0] + b" \t ",
        arch=ARCHS[0],
        opts="-fgpu-rdc -O3",
        clean_str_key_components=False,
    ) != linkercache._make_cache_key(
        buffer=BUFS[0],
        arch=ARCHS[0] + "     \t",
        opts="    -fgpu-rdc -O3  ",
        clean_str_key_components=False,
    )


def test_01_insert_get_delete_entry_for_buffer():
    assert len(linkercache._cache) == 0
    #
    for buf in BUFS:
        linkercache.get_or_insert_entry_for_buffer(
            buffer=buf,
            arch=ARCHS[0],
            opts=OPTS[0],
            entry=ENTRIES[0],  # inserts because entry is specified
            clean_str_key_components=True,  # remove whitespace from str keys
        )

    for buf in BUFS:
        linkercache.get_or_insert_entry_for_buffer(
            buffer=buf,
            arch=ARCHS[0],
            opts=OPTS[0] + " \t\n  \t\n   \t ",
            entry=ENTRIES[0],  # inserts because entry is specified
            clean_str_key_components=True,  # remove whitespace from str keys
        )

    assert len(linkercache._cache) == 2

    for buf in BUFS:
        entry = linkercache.get_or_insert_entry_for_buffer(
            buffer=buf,
            arch=ARCHS[0],
            opts=OPTS[0],
        )  # returns entry because entry is not specified/None
        assert entry == ENTRIES[0]
        linkercache.delete_entry_for_buffer(
            buffer=buf,
            arch=ARCHS[0],
            opts=OPTS[0],
        )

    assert len(linkercache._cache) == 0


def test_02_insert_get_delete_entry_for_file():
    assert len(linkercache._cache) == 0
    #
    files = []
    for buf in BUFS:
        fd, tmp = tempfile.mkstemp()
        files.append(tmp)
        os.close(fd)
        write_mode = "w" if isinstance(buf, str) else "wb"
        with open(tmp, write_mode) as outfile:
            outfile.write(buf)

    for filepath in files:
        linkercache.get_or_insert_entry_for_file(
            filepath=filepath,
            arch=ARCHS[0],
            opts=OPTS[0],
            entry=ENTRIES[0],  # inserts because entry is specified
            clean_str_key_components=True,  # remove whitespace from str keys
        )

    for filepath in files:
        linkercache.get_or_insert_entry_for_file(
            filepath=filepath,
            arch=ARCHS[0],
            opts=OPTS[0] + " \t\n  \t\n   \t ",
            entry=ENTRIES[0],  # inserts because entry is specified
            clean_str_key_components=True,  # remove whitespace from str keys
        )

    assert len(linkercache._cache) == 2

    for filepath in files:
        entry = linkercache.get_or_insert_entry_for_file(
            filepath=filepath,
            arch=ARCHS[0],
            opts=OPTS[0],
        )  # returns entry because entry is not specified/None
        assert entry == ENTRIES[0]
        linkercache.delete_entry_for_file(
            filepath=filepath,
            arch=ARCHS[0],
            opts=OPTS[0],
        )

    assert len(linkercache._cache) == 0
