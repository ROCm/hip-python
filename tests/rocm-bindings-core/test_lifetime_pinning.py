# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Regression tests for the program-lifetime intern dicts on
``rocm.bindings.util.types.CStr`` and ``ListOfBytes``.

Locks down the lifetime contract: when a caller passes a ``str`` /
``bytes`` (or list of those) to one of these wrapper classes, the
canonical encoded ``bytes`` is registered in a class-level intern
dict and stays alive for the program's lifetime. The ``void*`` /
``const char*`` handed to the backend therefore remains valid even
after:

  * the wrapper instance is collected, AND
  * the source python binding (e.g. a local list of bytes literals
    in the caller) is dropped, AND
  * the GC has churned the heap.

This addresses the failure mode where the backend (e.g.
``hiprtcCompileProgram`` via the COMGR compile cache) retains the
caller-supplied pointer past the python call's return; without the
intern, the backend's later read dereferences freed memory.

Tests also lock down:

  * ``str`` inputs are now accepted (``CStr`` previously raised on
    them; ``ListOfBytes`` previously rejected anything outside
    ``bytes`` / ``CStr``).
  * Identical content via ``str`` and via ``bytes`` paths produces
    the same canonical C pointer (intern-by-content).
  * The intern dict is bounded by the number of *unique* string
    contents ever passed, not by call count.
  * The private ``_clear_retained_inputs()`` test-only escape hatch
    empties both class dicts.
"""

import ctypes
import gc

import pytest
from rocm.bindings.util import types as _t


def _ptr_value(wrapper):
    """Read the C pointer out of a wrapper as an int."""
    return int(wrapper)


def _read_cstring(wrapper):
    """Read the underlying NUL-terminated bytes from the wrapper."""
    addr = _ptr_value(wrapper)
    if addr == 0:
        return b""
    return ctypes.string_at(addr)


def _churn_heap():
    """Force GC + a few churn allocations so any UAF would surface."""
    gc.collect()
    # Fill and drop a few intermediate buffers — increases the chance
    # that a freed bytes object's storage gets reused (so a stale
    # pointer would now read garbage instead of the lucky still-intact
    # original content).
    junk = [bytearray(b"X" * 1024) for _ in range(64)]
    del junk
    gc.collect()


@pytest.fixture(autouse=True)
def _clean_intern_dicts():
    """Reset the intern dicts before each test so dedup-bound
    assertions get a deterministic baseline. The escape hatch is
    private (underscore-prefixed) — only this regression suite
    should call it."""
    _t._clear_retained_inputs()
    yield
    _t._clear_retained_inputs()


# ---------------------------------------------------------------------------
# CStr
# ---------------------------------------------------------------------------


def test_cstr_str_accepted_and_pointer_reads_back_utf8():
    """`CStr` previously raised on `str` inputs; now it should
    UTF-8-encode and intern."""
    s = "hello-utf8-π"  # non-ASCII to confirm UTF-8 encoding
    w = _t.CStr(s)
    assert _ptr_value(w) != 0
    assert _read_cstring(w) == s.encode("utf-8")


def test_cstr_bytes_pointer_survives_source_drop_and_wrapper_drop():
    """The canonical bytes is pinned in the intern dict for the
    program's lifetime — dropping both the input local AND the
    wrapper instance must not invalidate the C pointer."""
    src = b"hello-cstr-survives"
    w = _t.CStr(src)
    addr = _ptr_value(w)
    # Drop both source and wrapper. The canonical bytes still lives
    # in the intern dict.
    del src, w
    _churn_heap()
    assert ctypes.string_at(addr) == b"hello-cstr-survives"


def test_cstr_str_pointer_survives_source_drop_and_wrapper_drop():
    """Same as above but via the new `str` path — the encoded
    canonical bytes is the one that's pinned."""
    s = "hello-cstr-str-survives"
    w = _t.CStr(s)
    addr = _ptr_value(w)
    del s, w
    _churn_heap()
    assert ctypes.string_at(addr) == b"hello-cstr-str-survives"


def test_cstr_str_and_bytes_dedup_to_same_canonical_pointer():
    """Identical logical content via `str` and `bytes` paths produces
    the same canonical C pointer — the intern table dedupes by
    encoded-content, regardless of input type."""
    a = _t.CStr("dedup-me")
    b = _t.CStr(b"dedup-me")
    assert _ptr_value(a) == _ptr_value(b)


def test_cstr_intern_dedup_bound_for_repeated_identical_input():
    """1000 invocations with identical content must keep the intern
    dict at exactly 1 entry."""
    for _ in range(1000):
        _t.CStr(b"same-content")
    assert len(_t.CStr._retained_inputs) == 1


def test_cstr_intern_grows_linearly_for_unique_inputs():
    """1000 invocations with unique content must grow the intern
    dict to exactly 1000 entries — bound is unique-input count, not
    call count."""
    for i in range(1000):
        _t.CStr(f"unique-{i}".encode("utf-8"))
    assert len(_t.CStr._retained_inputs) == 1000


# ---------------------------------------------------------------------------
# ListOfBytes
# ---------------------------------------------------------------------------


def test_listofbytes_str_entries_accepted():
    """`ListOfBytes` previously raised on `str` entries; now they
    should UTF-8-encode and intern."""
    w = _t.ListOfBytes(["alpha", "beta", "γ"])
    snapshot = _snapshot_inner_ptrs(w, 3)
    assert ctypes.string_at(snapshot[0]) == b"alpha"
    assert ctypes.string_at(snapshot[1]) == b"beta"
    assert ctypes.string_at(snapshot[2]) == "γ".encode("utf-8")


def _snapshot_inner_ptrs(w, n):
    """Read the inner ``char*`` entries out of the wrapper's ``void**``
    array as a list of int addresses. This simulates what a C library
    does during the call — it copies the pointers into its own state.
    After this snapshot, the wrapper's ``void**`` array is no longer
    needed; only the pointed-to canonical bytes need to outlive it."""
    addr = _ptr_value(w)
    arr = (ctypes.c_void_p * n).from_address(addr)
    return [int(arr[i] or 0) for i in range(n)]


def test_listofbytes_bytes_entries_pin_for_program_lifetime():
    """Build the wrapper from a literal-temporary list of bytes,
    snapshot the inner char* pointers (mimicking what a C library does
    during the call), drop both the source binding AND the wrapper,
    churn the heap, and confirm the snapshotted pointers still resolve
    to the original content. The wrapper-owned ``void**`` array is
    freed when the wrapper dies (correct — the C library has its own
    pointer copies by then); the canonical bytes that those pointers
    address must survive via the intern dict."""
    src = [b"--offload-arch=gfx90a", b"-O3", b"-Wall"]
    w = _t.ListOfBytes(src)
    snapshot = _snapshot_inner_ptrs(w, 3)
    del src, w
    _churn_heap()
    assert ctypes.string_at(snapshot[0]) == b"--offload-arch=gfx90a"
    assert ctypes.string_at(snapshot[1]) == b"-O3"
    assert ctypes.string_at(snapshot[2]) == b"-Wall"


def test_listofbytes_str_entries_pin_for_program_lifetime():
    """Same shape as the bytes test but via the `str` entry path.
    The encoded UTF-8 bytes is the canonical retained object."""
    w = _t.ListOfBytes(["--offload-arch=gfx90a", "-O3"])
    snapshot = _snapshot_inner_ptrs(w, 2)
    del w
    _churn_heap()
    assert ctypes.string_at(snapshot[0]) == b"--offload-arch=gfx90a"
    assert ctypes.string_at(snapshot[1]) == b"-O3"


def test_listofbytes_str_and_bytes_entries_dedup_to_same_pointer():
    """An entry passed once as `str` and once as the equivalent
    `bytes` ends up with the same canonical C pointer."""
    a = _t.ListOfBytes(["dedup-me"])
    b = _t.ListOfBytes([b"dedup-me"])
    a_inner = _snapshot_inner_ptrs(a, 1)[0]
    b_inner = _snapshot_inner_ptrs(b, 1)[0]
    # The void** arrays themselves are different (each wrapper owns
    # its own malloc), but the inner char* must be identical (both
    # point at the same canonical bytes in the intern dict).
    assert a_inner == b_inner


def test_listofbytes_intern_dedup_bound_for_repeated_identical_input():
    """1000 invocations with identical entries → intern stays at 1."""
    for _ in range(1000):
        _t.ListOfBytes([b"same-flag"])
    assert len(_t.ListOfBytes._retained_inputs) == 1


def test_listofbytes_intern_grows_linearly_for_unique_inputs():
    """1000 invocations with unique entries → intern reaches 1000."""
    for i in range(1000):
        _t.ListOfBytes([f"unique-{i}".encode("utf-8")])
    assert len(_t.ListOfBytes._retained_inputs) == 1000


def test_listofbytes_cstr_entries_pin_the_cstr_instance():
    """When an entry is a `CStr` instance, the CStr itself must be
    pinned in the intern dict (not just its pointer-into-bytes
    pinned via `CStr._retained_inputs`) — that way both the wrapper
    and its underlying buffer survive the caller dropping the
    CStr local."""
    cstr = _t.CStr("hello-via-cstr")
    cstr_id = id(cstr)
    w = _t.ListOfBytes([cstr])
    snapshot = _snapshot_inner_ptrs(w, 1)
    del cstr, w
    _churn_heap()
    assert ctypes.string_at(snapshot[0]) == b"hello-via-cstr"
    # The CStr instance itself must still be in the intern dict
    # (key = the original CStr); confirm via id matching against any
    # entry's id.
    assert any(id(k) == cstr_id for k in _t.ListOfBytes._retained_inputs)


# ---------------------------------------------------------------------------
# _clear_retained_inputs (test-only escape hatch)
# ---------------------------------------------------------------------------


def test_clear_retained_inputs_empties_both_dicts():
    _t.CStr(b"populate-cstr")
    _t.ListOfBytes([b"populate-list"])
    assert len(_t.CStr._retained_inputs) >= 1
    assert len(_t.ListOfBytes._retained_inputs) >= 1
    _t._clear_retained_inputs()
    assert len(_t.CStr._retained_inputs) == 0
    assert len(_t.ListOfBytes._retained_inputs) == 0


# ---------------------------------------------------------------------------
# Backwards-compat smoke tests — old paths still work
# ---------------------------------------------------------------------------


def test_cstr_bytes_input_still_works():
    """Pre-existing bytes-input path stays functional."""
    w = _t.CStr(b"plain-bytes-input")
    assert _read_cstring(w) == b"plain-bytes-input"


def test_listofbytes_mixed_bytes_and_cstr_entries():
    """A list with mixed `bytes` and `CStr` entries is still
    accepted; pointers read back as expected."""
    w = _t.ListOfBytes([b"plain", _t.CStr("via-cstr")])
    snapshot = _snapshot_inner_ptrs(w, 2)
    assert ctypes.string_at(snapshot[0]) == b"plain"
    assert ctypes.string_at(snapshot[1]) == b"via-cstr"


def test_listofbytes_rejects_unsupported_entry_type():
    """Entries that aren't bytes/str/CStr still raise — though the
    error type is now `TypeError` (was `ValueError`)."""
    with pytest.raises(TypeError):
        _t.ListOfBytes([42])  # int — not supported
