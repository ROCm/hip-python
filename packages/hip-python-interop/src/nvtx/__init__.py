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

"""An ``nvtx`` (NVTX) compatibility shim for ROCm.

This is NOT a port of the upstream (Apache-2.0) ``nvtx`` Python package. It is a
fresh, MIT-licensed re-implementation of the ``nvtx`` Python surface, backed
entirely by ROCTX via the high-level :py:obj:`rocm.bindings.roctx` bindings.

Goal: code that does ``import nvtx`` and uses annotations, ranges and markers
keeps working on AMD GPUs without modification. Profile with a ROCm-aware tool
(e.g. ``rocprofv3``/``rocprof`` or Omnitrace) instead of Nsight Systems.

Implemented surface (faithfully backed by ROCTX):

* :func:`mark` - instantaneous event (``roctxMarkA``).
* :func:`push_range` / :func:`pop_range` - nested, per-thread ranges
  (``roctxRangePushA`` / ``roctxRangePop``).
* :func:`start_range` / :func:`end_range` - process ranges that may cross
  threads (``roctxRangeStartA`` / ``roctxRangeStop``).
* :class:`annotate` - decorator and context manager around push/pop ranges.
* :class:`Profile` - automatic function annotation via ``sys.setprofile`` /
  ``threading.setprofile``.
* :func:`enabled` - honors the ``NVTX_DISABLE`` environment variable and the
  availability of the ROCTX runtime.

.. important::

   **Limitations.** ROCTX is a message-only tracing API. It has no concept of
   domains, colors, categories, payloads, registered strings, or counters.
   The following parts of the ``nvtx`` API are therefore accepted for source
   compatibility but do **not** behave as they would on NVIDIA hardware:

   Accepted but silently *dropped* (only the message reaches ROCTX):

   * ``domain`` / :func:`get_domain` / :class:`Domain` - ROCTX has no domain
     concept. Domain objects route to the same global ROCTX calls; the domain
     name has no effect and cross-domain isolation is lost. All events share a
     single per-thread namespace.
   * ``color`` (and :func:`nvtx.colors.color_to_hex`) - no color channel.
   * ``category`` / :func:`Domain.get_category_id` - no category channel.
   * ``payload`` - no payload channel.
   * :class:`RegisteredString` / :func:`Domain.get_registered_string`,
     :class:`EventAttributes` / :func:`Domain.get_event_attributes` /
     :func:`Domain.set_event_attributes` - lightweight holders only; the
     message is re-sent as a plain string on every call.

   Accepted but complete *no-ops* (record nothing):

   * :class:`Counter`, :class:`Int64Counter`, :class:`Float64Counter`,
     :class:`ExtCounter`, :func:`Domain.get_counter`, ``sample``,
     ``sample_no_value``, ``batch_submit``, :func:`Domain.get_timestamp`.
   * :class:`CounterSemantics` and the enums :class:`CounterValueType`,
     :class:`CounterInterpolation`, :class:`CounterNoValueReason`,
     :class:`TimestampType` (defined for import compatibility only).
   * :func:`numpy_dtype` (builds a NumPy dtype but carries no runtime effect).

   The counter/semantics surface additionally postdates NVTX release-v3 and is
   provided here only as forward-compatible stubs.
"""

import enum
import os
import sys
import threading
import time
from functools import lru_cache, wraps

from . import colors

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

__all__ = [
    "annotate",
    "enabled",
    "mark",
    "push_range",
    "pop_range",
    "start_range",
    "end_range",
    "get_domain",
    "Domain",
    "DummyDomain",
    "Profile",
    "EventAttributes",
    "RegisteredString",
    "Counter",
    "Int64Counter",
    "Float64Counter",
    "ExtCounter",
    "DummyCounter",
    "CounterSemantics",
    "CounterValueType",
    "CounterInterpolation",
    "CounterNoValueReason",
    "TimestampType",
    "numpy_dtype",
    "colors",
]

# ---------------------------------------------------------------------------
# ROCTX backend wiring
# ---------------------------------------------------------------------------

try:
    from rocm.bindings import roctx as _roctx
except Exception:  # pragma: no cover - roctx bindings not importable
    _roctx = None


def _roctx_runtime_available():
    """Probe whether the ROCTX runtime can actually be reached."""
    if _roctx is None:
        return False
    try:
        _roctx.roctx_version_major()
        return True
    except Exception:  # pragma: no cover - runtime library missing
        return False


# Mirror upstream ``nvtx``: any value of ``NVTX_DISABLE`` disables annotations.
# We additionally require the ROCTX runtime to be reachable.
_ENABLED = (not os.getenv("NVTX_DISABLE", False)) and _roctx_runtime_available()


_DONT_SET = object()


def _to_message(message):
    return "" if message is None else message


def _push(message):
    if _ENABLED:
        _roctx.roctxRangePushA(_to_message(message))


def _pop():
    if _ENABLED:
        _roctx.roctxRangePop()


def _mark(message):
    if _ENABLED:
        _roctx.roctxMarkA(_to_message(message))


def _start(message):
    if _ENABLED:
        return _roctx.roctxRangeStartA(_to_message(message))[0]
    return 0


def _stop(range_id):
    if _ENABLED:
        _roctx.roctxRangeStop(range_id)


def _unwrap_range_id(range_id):
    if isinstance(range_id, (tuple, list)):
        return range_id[0]
    return range_id


def enabled():
    """Return ``True`` if NVTX annotations are enabled.

    Annotations are disabled when the ``NVTX_DISABLE`` environment variable is
    set (to any value) or when the ROCTX runtime is not available.
    """
    return _ENABLED


# ---------------------------------------------------------------------------
# Module-level API (faithfully backed by ROCTX)
# ---------------------------------------------------------------------------


class annotate:
    """Annotate code ranges using a context manager or a decorator.

    Args:
        message: Message for the range; as a decorator it defaults to the decorated function name, as a context manager to the empty string.
        color: Accepted for compatibility; **dropped** (ROCTX has no color).
        domain: Accepted for compatibility; **dropped** (ROCTX has no domains).
        category: Accepted for compatibility; **dropped**.
        payload: Accepted for compatibility; **dropped**.

    Examples:
        Using a decorator (``message`` defaults to ``"func"``)::

            @nvtx.annotate(color="red", domain="my_domain")
            def func():
                ...

        Using a context manager::

            with nvtx.annotate("my_code_range", color="blue"):
                ...
    """

    def __init__(
        self,
        message=None,
        color=None,
        domain=None,
        category=None,
        payload=None,
    ):
        self.init_args = (message, color, domain, category, payload)
        self.message = message

    def __reduce__(self):
        return self.__class__, self.init_args

    def __enter__(self):
        _push(self.message)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _pop()
        return False

    def __call__(self, func):
        message = self.message if self.message is not None else func.__name__

        @wraps(func)
        def inner(*args, **kwargs):
            _push(message)
            try:
                return func(*args, **kwargs)
            finally:
                # Always pop the range, even if an exception is raised.
                _pop()

        return inner


def mark(message=None, color="blue", domain=None, category=None, payload=None):
    """Mark an instantaneous event.

    Args:
        message: A message associated with the event.
        color: Accepted for compatibility; **dropped** (ROCTX has no color).
        domain: Accepted for compatibility; **dropped** (ROCTX has no domains).
        category: Accepted for compatibility; **dropped**.
        payload: Accepted for compatibility; **dropped**.
    """
    _mark(message)


def push_range(
    message=None, color="blue", domain=None, category=None, payload=None
):
    """Mark the beginning of a (nested, per-thread) code range.

    Args:
        message: A message associated with the annotated code range.
        color: Accepted for compatibility; **dropped** (ROCTX has no color).
        domain: Accepted for compatibility; **dropped** (ROCTX has no domains).
        category: Accepted for compatibility; **dropped**.
        payload: Accepted for compatibility; **dropped**.

    Note:
        When applicable, prefer :class:`annotate`.
    """
    _push(message)


def pop_range(domain=None):
    """Mark the end of a code range started with :func:`push_range`.

    Args:
        domain: Accepted for compatibility; **dropped** (ROCTX has no domains).
    """
    _pop()


def start_range(
    message=None, color=None, domain=None, category=None, payload=None
):
    """Mark the beginning of a process range.

    Args:
        message: A message associated with the range.
        color: Accepted for compatibility; **dropped** (ROCTX has no color).
        domain: Accepted for compatibility; **dropped** (ROCTX has no domains).
        category: Accepted for compatibility; **dropped**.
        payload: Accepted for compatibility; **dropped**.

    Returns:
        A ``(range_id, domain_handle)`` tuple that must be passed to
        :func:`end_range`. ``domain_handle`` is always ``0`` in this shim.
    """
    return (_start(message), 0)


def end_range(range_id):
    """Mark the end of a process range started with :func:`start_range`.

    Args:
        range_id: The tuple (or bare id) returned by :func:`start_range`.
    """
    if range_id is None:
        return
    _stop(_unwrap_range_id(range_id))


# ---------------------------------------------------------------------------
# Profiler (automatic function annotation)
# ---------------------------------------------------------------------------


class Profile:
    """Programmatically control NVTX automatic annotations.

    Wraps every Python (and, optionally, C-extension) function call in a ROCTX
    range using ``sys.setprofile`` / ``threading.setprofile``.

    Args:
        linenos: Include file and line number information in annotations.
        annotate_cfuncs: Also annotate C-extension and builtin functions.

    Examples:
        >>> import nvtx, time
        >>> pr = nvtx.Profile()
        >>> pr.enable()
        >>> time.sleep(1)  # captured by nvtx
        >>> pr.disable()
        >>> time.sleep(1)  # not captured
    """

    def __init__(self, linenos=True, annotate_cfuncs=True):
        self.linenos = linenos
        self.annotate_cfuncs = annotate_cfuncs

    def _profile(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            if self.linenos:
                fname = os.path.basename(frame.f_code.co_filename)
                message = f"{fname}:{frame.f_lineno}({name})"
            else:
                message = name
            _push(message)
        elif event == "c_call" and self.annotate_cfuncs:
            _push(getattr(arg, "__name__", "<c_call>"))
        elif event == "return":
            _pop()
        elif event in ("c_return", "c_exception") and self.annotate_cfuncs:
            _pop()
        return None

    def enable(self):
        """Start annotating function calls automatically."""
        if _ENABLED:
            threading.setprofile(self._profile)
            sys.setprofile(self._profile)

    def disable(self):
        """Stop annotating function calls automatically."""
        if _ENABLED:
            sys.setprofile(None)
            threading.setprofile(None)


# ---------------------------------------------------------------------------
# Degraded surface: registered strings / event attributes (message-only)
# ---------------------------------------------------------------------------


class RegisteredString:
    """A stand-in for an NVTX registered string.

    ROCTX has no string-registration API, so this simply wraps the raw string;
    the message is re-sent as a plain string on every event.
    """

    def __init__(self, domain=None, string=None):
        self.domain = domain
        self.string = string

    def __str__(self):
        return "" if self.string is None else str(self.string)


class EventAttributes:
    """A holder for event attributes.

    Only :attr:`message` reaches ROCTX; ``color``, ``category`` and ``payload``
    are stored for compatibility but **dropped**.
    """

    def __init__(
        self, domain=None, message=None, color=None, category=None, payload=None
    ):
        self.domain = domain
        self.message = message
        self.color = color
        self.category = category
        self.payload = payload


def _message_text(attributes, kwargs):
    """Resolve the message string from attributes or keyword arguments."""
    if attributes is not None:
        message = getattr(attributes, "message", None)
    else:
        message = kwargs.get("message")
    if isinstance(message, RegisteredString):
        return message.string
    return message


# ---------------------------------------------------------------------------
# Degraded surface: domains
# ---------------------------------------------------------------------------


class Domain:
    """A per-domain interface to the NVTX API.

    .. important::

       ROCTX has no domain concept. The ``name`` is accepted but ignored, and
       every method routes to the same global ROCTX calls, so events from
       different :class:`Domain` instances are **not** isolated from one another.
    """

    def __init__(self, name=None):
        self.name = name
        # ROCTX has no domain handle; kept for API compatibility only.
        self.handle = 0
        self._categories = {}

    def push_range(self, attributes=None, **kwargs):
        """Mark the beginning of a code range (see :func:`push_range`)."""
        _push(_message_text(attributes, kwargs))

    def pop_range(self):
        """Mark the end of a code range (see :func:`pop_range`)."""
        _pop()

    def mark(self, attributes=None, **kwargs):
        """Mark an instantaneous event (see :func:`mark`)."""
        _mark(_message_text(attributes, kwargs))

    def start_range(self, attributes=None, **kwargs):
        """Mark the beginning of a process range (see :func:`start_range`)."""
        return _start(_message_text(attributes, kwargs))

    def end_range(self, range_id):
        """Mark the end of a process range (see :func:`end_range`)."""
        _stop(_unwrap_range_id(range_id))

    def get_category_id(self, name):
        """Return a synthetic category id (not honored by ROCTX)."""
        return self._categories.setdefault(name, len(self._categories) + 1)

    def get_registered_string(self, string):
        """Return a :class:`RegisteredString` wrapper (no C registration)."""
        return RegisteredString(self, string)

    def get_event_attributes(
        self, message=None, color=None, category=None, payload=None
    ):
        """Create an :class:`EventAttributes` object."""
        return EventAttributes(self, message, color, category, payload)

    def set_event_attributes(
        self,
        attributes,
        *,
        message=_DONT_SET,
        color=_DONT_SET,
        category=_DONT_SET,
        payload=_DONT_SET,
    ):
        """Set attributes on an existing :class:`EventAttributes` object."""
        if message is not _DONT_SET:
            attributes.message = message
        if color is not _DONT_SET:
            attributes.color = color
        if category is not _DONT_SET:
            attributes.category = category
        if payload is not _DONT_SET:
            attributes.payload = payload

    def get_timestamp(self):
        """Return a monotonic timestamp (for API compatibility only)."""
        return time.perf_counter_ns()

    def get_counter(
        self,
        name,
        dtype,
        *,
        description=None,
        scope=None,
        semantics=None,
        time_domain=None,
    ):
        """Return a no-op counter (ROCTX has no counter API)."""
        if dtype is int:
            return Int64Counter(self, name)
        if dtype is float:
            return Float64Counter(self, name)
        return ExtCounter(self, name, dtype)


class DummyDomain:
    """A no-op replacement for :class:`Domain` used when NVTX is disabled."""

    handle = 0
    name = None

    def push_range(self, attributes=None, **kwargs):
        pass

    def pop_range(self):
        pass

    def mark(self, attributes=None, **kwargs):
        pass

    def start_range(self, attributes=None, **kwargs):
        return 0

    def end_range(self, range_id):
        pass

    def get_category_id(self, name):
        return 0

    def get_registered_string(self, string):
        return RegisteredString(self, string)

    def get_event_attributes(
        self, message=None, color=None, category=None, payload=None
    ):
        return EventAttributes(self, message, color, category, payload)

    def set_event_attributes(self, attributes, **kwargs):
        pass

    def get_timestamp(self):
        return 0

    def get_counter(self, name, dtype, **kwargs):
        return DummyCounter()


dummy_domain = DummyDomain()


@lru_cache(maxsize=None)
def _get_domain_cached(name):
    return Domain(name)


def get_domain(name=None):
    """Get or create a :class:`Domain` for a domain name.

    Returns a :class:`DummyDomain` when NVTX is disabled.
    """
    if not _ENABLED:
        return dummy_domain
    return _get_domain_cached(name)


# ---------------------------------------------------------------------------
# No-op surface: counters, semantics and enums
# ---------------------------------------------------------------------------


class Counter:
    """Base class for NVTX counters.

    .. important::

       ROCTX has no counter API, so all counter operations in this shim are
       **no-ops** that record nothing.
    """

    def __init__(self, domain=None, name=None, dtype=None):
        self.domain = domain
        self.name = name
        self.dtype = dtype

    def sample(self, value):
        """Record one counter sample (no-op)."""

    def sample_no_value(self, reason):
        """Record a missing counter sample (no-op)."""

    def batch_submit(self, values, timestamps):
        """Record a batch of counter samples (no-op)."""


class Int64Counter(Counter):
    """Counter for signed 64-bit integer samples (no-op)."""

    def __init__(self, domain=None, name=None):
        super().__init__(domain, name, int)


class Float64Counter(Counter):
    """Counter for double-precision floating-point samples (no-op)."""

    def __init__(self, domain=None, name=None):
        super().__init__(domain, name, float)


class ExtCounter(Counter):
    """Counter for NumPy dtype-based samples and counter groups (no-op)."""


class DummyCounter(Counter):
    """A no-op replacement for :class:`Counter` when the domain is disabled."""


class CounterSemantics:
    """Metadata describing how a counter value should be interpreted.

    Stored for API compatibility only; it has no runtime effect on ROCTX.
    """

    def __init__(
        self,
        unit=None,
        value_type=None,
        interpolation=None,
        min=None,
        max=None,
        unit_scale_numerator=1,
        unit_scale_denominator=1,
    ):
        self.unit = unit
        self.value_type = (
            value_type if value_type is not None else CounterValueType.ABSOLUTE
        )
        self.interpolation = (
            interpolation
            if interpolation is not None
            else CounterInterpolation.POINT
        )
        self.min = min
        self.max = max
        self.unit_scale_numerator = unit_scale_numerator
        self.unit_scale_denominator = unit_scale_denominator


class CounterValueType(enum.IntEnum):
    """How counter sample values relate to previous samples."""

    ABSOLUTE = 0
    DELTA = 1
    DELTA_SINCE_START = 2


class CounterInterpolation(enum.IntEnum):
    """How tools should interpolate counter values between samples."""

    POINT = 0
    SINCE_LAST = 1
    UNTIL_NEXT = 2
    LINEAR = 3


class CounterNoValueReason(enum.IntEnum):
    """Reasons for recording a counter sample without a value."""

    ZERO = 0
    UNCHANGED = 1
    UNAVAILABLE = 2


class TimestampType(enum.IntEnum):
    """Timestamp domains that can be associated with batched samples."""

    NONE = 0
    TOOL_PROVIDED = 1
    CPU_TSC = 2
    CPU_CLOCK_GETTIME_REALTIME = 3
    CPU_CLOCK_GETTIME_MONOTONIC = 4
    GPU_GLOBALTIMER = 5


def numpy_dtype(*args, counter_semantics=None, **kwargs):
    """Construct a NumPy dtype, optionally carrying counter semantics metadata.

    Accepts the same arguments as :func:`numpy.dtype`. The ``counter_semantics``
    metadata is attached but has **no runtime effect** on ROCTX.

    Raises:
        RuntimeError: If NumPy is not installed.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "numpy_dtype requires NumPy, which is not installed."
        ) from e
    dtype = np.dtype(*args, **kwargs)
    if counter_semantics is not None:
        metadata = dict(dtype.metadata or {})
        metadata.setdefault("nvtx", counter_semantics)
        dtype = np.dtype(dtype, metadata=metadata)
    return dtype
