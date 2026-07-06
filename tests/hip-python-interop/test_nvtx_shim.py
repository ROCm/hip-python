#!/usr/bin/env -S python3 -m pytest -v -s
# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Unit tests for the hip-python-interop ``nvtx`` (NVTX) compatibility shim.

These exercise the GPU-independent code paths of the shim by stubbing the
``rocm.bindings.roctx`` calls it dispatches to, so they run without a GPU (or a
ROCTX runtime) present.

The shim lives inside the installed ``hip_python_interop`` wheel, not in this
tree, so the test imports the installed top-level ``nvtx`` package.
"""

__author__ = "Advanced Micro Devices, Inc."

import sys
import types

import pytest

# Skip the whole module when the interop wheel is not importable, rather than
# erroring at collection time.
nvtx = pytest.importorskip("nvtx")


class _Recorder:
    """A fake ``rocm.bindings.roctx`` module that records dispatched calls."""

    def __init__(self):
        self.calls = []

    def roctx_version_major(self):
        return (4,)

    def roctxMarkA(self, message):
        self.calls.append(("mark", message))

    def roctxRangePushA(self, message):
        self.calls.append(("push", message))
        return (0,)

    def roctxRangePop(self):
        self.calls.append(("pop",))
        return (0,)

    def roctxRangeStartA(self, message):
        self.calls.append(("start", message))
        return (7,)

    def roctxRangeStop(self, range_id):
        self.calls.append(("stop", range_id))


@pytest.fixture
def rec(monkeypatch):
    """Enable the shim and route ROCTX calls into a recorder."""
    recorder = _Recorder()
    monkeypatch.setattr(nvtx, "_roctx", recorder, raising=False)
    monkeypatch.setattr(nvtx, "_ENABLED", True, raising=False)
    return recorder


def test_mark_dispatches_message(rec):
    nvtx.mark("checkpoint", color="green", domain="ignored")
    assert rec.calls == [("mark", "checkpoint")]


def test_push_pop_dispatch(rec):
    nvtx.push_range("region", color="blue")
    nvtx.pop_range(domain="ignored")
    assert rec.calls == [("push", "region"), ("pop",)]


def test_start_end_range_roundtrip(rec):
    range_id = nvtx.start_range("proc")
    assert range_id == (7, 0)
    nvtx.end_range(range_id)
    assert rec.calls == [("start", "proc"), ("stop", 7)]


def test_end_range_accepts_bare_int(rec):
    nvtx.end_range(7)
    assert rec.calls == [("stop", 7)]


def test_end_range_none_is_noop(rec):
    nvtx.end_range(None)
    assert rec.calls == []


def test_annotate_context_manager(rec):
    with nvtx.annotate("phase", color="red"):
        pass
    assert rec.calls == [("push", "phase"), ("pop",)]


def test_annotate_decorator_defaults_to_func_name(rec):
    @nvtx.annotate()
    def my_func():
        return 42

    assert my_func() == 42
    assert rec.calls == [("push", "my_func"), ("pop",)]


def test_annotate_decorator_pops_on_exception(rec):
    @nvtx.annotate("boom")
    def raiser():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        raiser()
    assert rec.calls == [("push", "boom"), ("pop",)]


def test_none_message_becomes_empty_string(rec):
    nvtx.mark()
    assert rec.calls == [("mark", "")]


def test_domain_routes_to_global_roctx(rec):
    # ROCTX has no domains: different domain names share one namespace.
    d1 = nvtx.get_domain("a")
    d2 = nvtx.get_domain("b")
    d1.push_range(message="x")
    d2.pop_range()
    d1.mark(message="m")
    rid = d1.start_range(message="s")
    d2.end_range(rid)
    assert rec.calls == [
        ("push", "x"),
        ("pop",),
        ("mark", "m"),
        ("start", "s"),
        ("stop", 7),
    ]


def test_domain_uses_event_attributes_message(rec):
    d = nvtx.get_domain("dom")
    attrs = d.get_event_attributes(message="attr-msg", color="red", category="c")
    d.push_range(attrs)
    d.pop_range()
    assert rec.calls == [("push", "attr-msg"), ("pop",)]


def test_profile_profile_hook_pushes_and_pops(rec):
    pr = nvtx.Profile(linenos=False)
    frame = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_name="fn", co_filename="/x/y.py"),
        f_lineno=10,
    )
    pr._profile(frame, "call", None)
    pr._profile(frame, "return", None)
    assert rec.calls == [("push", "fn"), ("pop",)]


def test_profile_enable_disable_toggles_hook(rec):
    assert sys.getprofile() is None
    pr = nvtx.Profile(linenos=False)
    pr.enable()
    try:
        # Bound methods compare equal but are re-created on each access, so
        # use ``==`` rather than ``is`` here.
        assert sys.getprofile() == pr._profile
    finally:
        pr.disable()
    assert sys.getprofile() is None


def test_counters_are_noops(rec):
    d = nvtx.get_domain("dom")
    int_counter = d.get_counter("bytes", int)
    float_counter = d.get_counter("loss", float)
    assert isinstance(int_counter, nvtx.Int64Counter)
    assert isinstance(float_counter, nvtx.Float64Counter)
    # No exceptions and nothing dispatched to ROCTX.
    int_counter.sample(4096)
    float_counter.sample(0.42)
    float_counter.sample_no_value(nvtx.CounterNoValueReason.UNAVAILABLE)
    int_counter.batch_submit([1, 2], [10, 20])
    assert rec.calls == []


def test_enums_expose_documented_members():
    assert nvtx.CounterValueType.ABSOLUTE == 0
    assert nvtx.CounterValueType.DELTA_SINCE_START == 2
    assert nvtx.CounterInterpolation.LINEAR == 3
    assert nvtx.CounterNoValueReason.UNAVAILABLE == 2
    assert nvtx.TimestampType.TOOL_PROVIDED == 1


def test_numpy_dtype_attaches_semantics():
    np = pytest.importorskip("numpy")
    semantics = nvtx.CounterSemantics(unit="bytes", min=0)
    dtype = nvtx.numpy_dtype("int64", counter_semantics=semantics)
    assert dtype == np.dtype("int64")
    assert dtype.metadata["nvtx"] is semantics


def test_disabled_shim_is_noop(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(nvtx, "_roctx", recorder, raising=False)
    monkeypatch.setattr(nvtx, "_ENABLED", False, raising=False)
    assert nvtx.enabled() is False
    nvtx.mark("x")
    nvtx.push_range("y")
    nvtx.pop_range()
    assert nvtx.start_range("z") == (0, 0)
    assert nvtx.get_domain("d") is nvtx.dummy_domain
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Compatibility mode
# ---------------------------------------------------------------------------


@pytest.fixture
def compat_mode():
    """Save/restore the compat mode and clear the domain cache around a test."""
    previous = nvtx.get_compat_mode()
    nvtx._get_domain_cached.cache_clear()
    try:
        yield nvtx.set_compat_mode
    finally:
        nvtx.set_compat_mode(previous)
        nvtx._get_domain_cached.cache_clear()


def test_default_compat_mode_is_silent():
    assert nvtx.get_compat_mode() == "silent"


def test_silent_mode_never_warns_or_raises(rec, compat_mode, recwarn):
    compat_mode("silent")
    nvtx.mark("m", color="red", domain="d", category="c", payload=1)
    nvtx.push_range("r", color="blue")
    nvtx.annotate("x", payload=5)
    nvtx.get_domain("named").get_category_id("cat")
    nvtx.get_domain("named2").get_counter("bytes", int)
    assert not any(
        isinstance(w.message, nvtx.NvtxCompatWarning) for w in recwarn.list
    )


@pytest.mark.parametrize(
    "action",
    [
        lambda: nvtx.mark("m", color="red"),
        lambda: nvtx.push_range("r", domain="d"),
        lambda: nvtx.pop_range(domain="d"),
        lambda: nvtx.start_range("s", category="c"),
        lambda: nvtx.annotate("x", payload=5),
        lambda: nvtx.get_domain("named"),
        lambda: nvtx.get_domain("named").get_category_id("cat"),
        lambda: nvtx.get_domain("named").get_counter("bytes", int),
    ],
)
def test_warn_mode_emits_warning(rec, compat_mode, action):
    compat_mode("warn")
    with pytest.warns(nvtx.NvtxCompatWarning):
        action()


@pytest.mark.parametrize(
    "action",
    [
        lambda: nvtx.mark("m", color="red"),
        lambda: nvtx.push_range("r", domain="d"),
        lambda: nvtx.pop_range(domain="d"),
        lambda: nvtx.start_range("s", category="c"),
        lambda: nvtx.annotate("x", payload=5),
        lambda: nvtx.get_domain("named"),
        lambda: nvtx.get_domain("named").get_category_id("cat"),
        lambda: nvtx.get_domain("named").get_counter("bytes", int),
    ],
)
def test_error_mode_raises(rec, compat_mode, action):
    compat_mode("error")
    with pytest.raises(nvtx.NvtxCompatError):
        action()


def test_domain_event_attributes_trigger_compat(rec, compat_mode):
    compat_mode("error")
    d = nvtx.get_domain()  # default (unnamed) domain: no compat on creation
    attrs = nvtx.EventAttributes(message="m", color="red")
    with pytest.raises(nvtx.NvtxCompatError):
        d.push_range(attrs)


def test_numpy_dtype_triggers_compat(compat_mode):
    pytest.importorskip("numpy")
    compat_mode("error")
    with pytest.raises(nvtx.NvtxCompatError):
        nvtx.numpy_dtype("int64")


def test_supported_calls_never_trigger_in_error_mode(rec, compat_mode):
    compat_mode("error")
    # None of these use an unsupported feature/argument.
    nvtx.mark("m")
    nvtx.push_range("r")
    nvtx.pop_range()
    rid = nvtx.start_range("s")
    nvtx.end_range(rid)
    with nvtx.annotate("ctx"):
        pass

    @nvtx.annotate()
    def fn():
        return 1

    assert fn() == 1
    pr = nvtx.Profile(linenos=False)
    pr.enable()
    pr.disable()
    d = nvtx.get_domain()  # default domain
    d.push_range(message="only-message")
    d.pop_range()
    # Explicit None for a dropped arg counts as "not supplied".
    nvtx.mark("m", color=None, domain=None)


def test_compat_fires_even_when_disabled(monkeypatch, compat_mode):
    monkeypatch.setattr(nvtx, "_ENABLED", False, raising=False)
    compat_mode("error")
    with pytest.raises(nvtx.NvtxCompatError):
        nvtx.mark("m", color="red")


def test_set_compat_mode_rejects_invalid():
    previous = nvtx.get_compat_mode()
    try:
        with pytest.raises(ValueError):
            nvtx.set_compat_mode("bogus")
    finally:
        nvtx.set_compat_mode(previous)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("silent", "silent"),
        ("warn", "warn"),
        ("error", "error"),
        ("WARN", "warn"),
        ("  Error  ", "error"),
        ("bogus", "silent"),
        (None, "silent"),
        ("", "silent"),
    ],
)
def test_parse_compat_mode(value, expected):
    assert nvtx._parse_compat_mode(value) == expected
