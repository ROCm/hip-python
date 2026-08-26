# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Tests for ``rocm.comgr.hiprtc_header``.

The header text COMGR needs in order to compile HIP source is read from the
installed ROCm's ``hiprtc-builtins`` library rather than shipped, because the
text encodes the data model of whichever host generated it. These tests cover
the reading, the failure modes, and the property that motivated the change:
that the text agrees with the platform it is going to be compiled for.
"""

import ctypes

import pytest
from rocm.comgr import hiprtc_header


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    """Resolution is cached for the process; each test starts from scratch."""
    monkeypatch.setattr(hiprtc_header, "_cached", None)
    monkeypatch.delenv(hiprtc_header.HEADER_PATH_ENV_VAR, raising=False)


needs_builtins = pytest.mark.skipif(
    hiprtc_header.find_builtins_library() is None,
    reason="no hiprtc-builtins library in this ROCm installation",
)


@needs_builtins
def test_reads_the_header_from_the_installation():
    text = hiprtc_header.get_hiprtc_runtime_header()

    assert hiprtc_header.get_hiprtc_runtime_header_origin() == str(
        hiprtc_header.find_builtins_library()
    )
    # Landmarks hipRTC's builtins have carried for as long as the file existed.
    assert "#pragma clang diagnostic push" in text
    assert "__ocml_" in text
    assert "__make_mantissa" in text


@needs_builtins
def test_size_t_matches_this_platform():
    """The defect this module exists to prevent.

    A header generated on an LP64 host declares ``size_t`` as ``long unsigned
    int``. Compiled for Windows, where that type is 32-bit, it both contradicts
    the compilation's own ``size_t`` and cannot hold what the header stores in
    it. So the text must not disagree with this platform's width.
    """
    text = hiprtc_header.get_hiprtc_runtime_header()

    typedefs = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("typedef")
        and line.strip().endswith("size_t;")
    ]
    assert typedefs, "the header always typedefs size_t"

    size_t_bits = ctypes.sizeof(ctypes.c_size_t) * 8
    long_bits = ctypes.sizeof(ctypes.c_long) * 8
    for typedef in typedefs:
        spelled_long = "long long" not in typedef
        if spelled_long and long_bits != size_t_bits:
            pytest.fail(
                f"{typedef!r} names a {long_bits}-bit type where size_t is "
                f"{size_t_bits}-bit: this header was generated for another "
                "data model"
            )


@needs_builtins
def test_line_endings_are_normalized():
    assert "\r" not in hiprtc_header.get_hiprtc_runtime_header()


@needs_builtins
def test_result_is_cached():
    first = hiprtc_header.get_hiprtc_runtime_header()
    # Make a second read impossible; a cached result is unaffected.
    hiprtc_header_find = hiprtc_header.find_builtins_library
    try:
        hiprtc_header.find_builtins_library = lambda: None
        assert hiprtc_header.get_hiprtc_runtime_header() is first
    finally:
        hiprtc_header.find_builtins_library = hiprtc_header_find


def test_env_var_overrides_the_installation(tmp_path, monkeypatch):
    header = tmp_path / "my_hiprtc_runtime.h"
    # Bytes, not text: text mode would translate the newline again on Windows.
    header.write_bytes(b"typedef unsigned long long size_t;\r\n")
    monkeypatch.setenv(hiprtc_header.HEADER_PATH_ENV_VAR, str(header))

    assert hiprtc_header.get_hiprtc_runtime_header() == (
        "typedef unsigned long long size_t;\n"
    )
    assert hiprtc_header.get_hiprtc_runtime_header_origin() == str(header)


def test_raises_when_no_library_is_found(monkeypatch):
    monkeypatch.setattr(hiprtc_header, "find_builtins_library", lambda: None)

    with pytest.raises(RuntimeError, match="could not locate"):
        hiprtc_header.get_hiprtc_runtime_header()


def test_raises_when_symbols_are_absent(tmp_path, monkeypatch):
    """A ROCm that renames the symbols must say so, not return a wrong header."""
    monkeypatch.setattr(
        hiprtc_header, "find_builtins_library", lambda: tmp_path / "fake"
    )
    monkeypatch.setattr(
        hiprtc_header, "read_header_from_library", lambda path: None
    )

    with pytest.raises(RuntimeError, match=hiprtc_header.HEADER_SYMBOL):
        hiprtc_header.get_hiprtc_runtime_header()


def test_unloadable_library_reports_no_header(tmp_path):
    """``read_header_from_library`` answers `None` instead of raising OSError."""
    not_a_library = tmp_path / "not_a_library.dll"
    not_a_library.write_bytes(b"certainly not a shared library")

    assert hiprtc_header.read_header_from_library(not_a_library) is None


@needs_builtins
def test_comgr_exposes_the_header_lazily():
    """``HIPRTC_RUNTIME_HEADER`` must not be resolved by importing the package.

    numba-hip and the COMGR examples read it as a module attribute, so the name
    has to keep working, but an import that loads hipRTC's builtins would make
    every other user of rocm.comgr pay for it.
    """
    import rocm.comgr
    import rocm.comgr.comgr

    if "HIPRTC_RUNTIME_HEADER" in vars(rocm.comgr.comgr):
        pytest.skip("something already resolved it in this process")

    assert rocm.comgr.HIPRTC_RUNTIME_HEADER.startswith(
        "#pragma clang diagnostic push"
    )
