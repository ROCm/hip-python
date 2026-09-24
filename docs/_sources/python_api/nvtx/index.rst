nvtx
====

.. py:module:: nvtx

.. autoapi-nested-parse::

   An ``nvtx`` (NVTX) compatibility shim for ROCm.

   This is NOT a port of the upstream (Apache-2.0) ``nvtx`` Python package. It is a
   fresh, MIT-licensed re-implementation of the ``nvtx`` Python surface, backed
   entirely by ROCTX via the high-level `~.rocm.bindings.roctx` bindings.

   Goal: code that does ``import nvtx`` and uses annotations, ranges and markers
   keeps working on AMD GPUs without modification. Profile with a ROCm-aware tool
   (e.g. ``rocprof-compute``) instead of Nsight Systems.

   Implemented surface (faithfully backed by ROCTX):

   * `~.mark` - instantaneous event (``roctxMarkA``).
   * `~.push_range` / `~.pop_range` - nested, per-thread ranges
     (``roctxRangePushA`` / ``roctxRangePop``).
   * `~.start_range` / `~.end_range` - process ranges that may cross
     threads (``roctxRangeStartA`` / ``roctxRangeStop``).
   * `~.annotate` - decorator and context manager around push/pop ranges.
   * `~.Profile` - automatic function annotation via ``sys.setprofile`` /
     ``threading.setprofile``.
   * `~.enabled` - honors the ``NVTX_DISABLE`` environment variable and the
     availability of the ROCTX runtime.

   .. important::

      **Limitations.** ROCTX is a message-only tracing API. It has no concept of
      domains, colors, categories, payloads, registered strings, or counters.
      The following parts of the ``nvtx`` API are therefore accepted for source
      compatibility but do **not** behave as they would on NVIDIA hardware:

      Accepted but silently *dropped* (only the message reaches ROCTX):

      * ``domain`` / `~.get_domain` / `~.Domain` - ROCTX has no domain
        concept. Domain objects route to the same global ROCTX calls; the domain
        name has no effect and cross-domain isolation is lost. All events share a
        single per-thread namespace.
      * ``color`` (and `~.colors.color_to_hex`) - no color channel.
      * ``category`` / `~.Domain.get_category_id` - no category channel.
      * ``payload`` - no payload channel.
      * `~.RegisteredString` / `~.Domain.get_registered_string`,
        `~.EventAttributes` / `~.Domain.get_event_attributes` /
        `~.Domain.set_event_attributes` - lightweight holders only; the
        message is re-sent as a plain string on every call.

      Accepted but complete *no-ops* (record nothing):

      * `~.Counter`, `~.Int64Counter`, `~.Float64Counter`,
        `~.ExtCounter`, `~.Domain.get_counter`, ``sample``,
        ``sample_no_value``, ``batch_submit``, `~.Domain.get_timestamp`.
      * `~.CounterSemantics` and the enums `~.CounterValueType`,
        `~.CounterInterpolation`, `~.CounterNoValueReason`,
        `~.TimestampType` (defined for import compatibility only).
      * `~.numpy_dtype` (builds a NumPy dtype but carries no runtime effect).

      The counter/semantics surface additionally postdates NVTX release-v3 and is
      provided here only as forward-compatible stubs.

   Compatibility mode
   ------------------

   To help audit whether code is portable to ROCTX, the shim can report when an
   unsupported feature is used or an unsupported (dropped) argument is supplied.
   The mode is one of:

   * ``"silent"`` (default) - accept and drop/no-op silently, preserving drop-in
     behavior.
   * ``"warn"`` - emit an `~.NvtxCompatWarning` and then drop/no-op.
   * ``"error"`` - raise an `~.NvtxCompatError`.

   Select it via the ``HIP_PYTHON_NVTX_COMPAT`` environment variable
   (``silent`` / ``warn`` / ``error``) or at runtime with
   `~.set_compat_mode` / `~.get_compat_mode`. The checks fire regardless
   of whether tracing is enabled (see `~.enabled`), so they also flag
   non-portable usage in CI that runs without a ROCTX runtime.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /python_api/nvtx/colors/index


Exceptions
----------

.. autoapisummary::

   nvtx.NvtxCompatWarning
   nvtx.NvtxCompatError


Classes
-------

.. autoapisummary::

   nvtx.annotate
   nvtx.Profile
   nvtx.RegisteredString
   nvtx.EventAttributes
   nvtx.Domain
   nvtx.DummyDomain
   nvtx.Counter
   nvtx.Int64Counter
   nvtx.Float64Counter
   nvtx.ExtCounter
   nvtx.DummyCounter
   nvtx.CounterSemantics
   nvtx.CounterValueType
   nvtx.CounterInterpolation
   nvtx.CounterNoValueReason
   nvtx.TimestampType


Functions
---------

.. autoapisummary::

   nvtx.get_compat_mode
   nvtx.set_compat_mode
   nvtx.enabled
   nvtx.mark
   nvtx.push_range
   nvtx.pop_range
   nvtx.start_range
   nvtx.end_range
   nvtx.get_domain
   nvtx.numpy_dtype


Package Contents
----------------

.. py:exception:: NvtxCompatWarning

   Bases: :py:obj:`UserWarning`


   Warning emitted when an unsupported NVTX feature/argument is used.

   Only emitted when the compatibility mode is ``"warn"`` (see
   `~.set_compat_mode`).


.. py:exception:: NvtxCompatError

   Bases: :py:obj:`RuntimeError`


   Error raised when an unsupported NVTX feature/argument is used.

   Only raised when the compatibility mode is ``"error"`` (see
   `~.set_compat_mode`).


.. py:function:: get_compat_mode()

   Return the current compatibility mode (``silent``/``warn``/``error``).


.. py:function:: set_compat_mode(mode)

   Set the compatibility mode.

   Args:
       mode:
           One of ``"silent"``, ``"warn"`` or ``"error"``.

   Raises:
       ``ValueError``:
           If ``mode`` is not a recognized compatibility mode.


.. py:function:: enabled()

   Return ``True`` if NVTX annotations are enabled.

   Annotations are disabled when the ``NVTX_DISABLE`` environment variable is
   set (to any value) or when the ROCTX runtime is not available.


.. py:class:: annotate(message=None, color=_UNSET, domain=_UNSET, category=_UNSET, payload=_UNSET)

   Annotate code ranges using a context manager or a decorator.

   Args:
       message:
           Message for the range; as a decorator it defaults to the decorated
           function name, as a context manager to the empty string.

       color:
           Accepted for compatibility; **dropped** (ROCTX has no color).

       domain:
           Accepted for compatibility; **dropped** (ROCTX has no domains).

       category:
           Accepted for compatibility; **dropped**.

       payload:
           Accepted for compatibility; **dropped**.

   Examples:
       Using a decorator (``message`` defaults to ``"func"``)::

           @nvtx.annotate(color="red", domain="my_domain")
           def func():
               ...

       Using a context manager::

           with nvtx.annotate("my_code_range", color="blue"):
               ...


   .. py:attribute:: init_args


   .. py:attribute:: message
      :value: None



.. py:function:: mark(message=None, color=_UNSET, domain=_UNSET, category=_UNSET, payload=_UNSET)

   Mark an instantaneous event.

   Args:
       message:
           A message associated with the event.

       color:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       domain:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       category:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       payload:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).


.. py:function:: push_range(message=None, color=_UNSET, domain=_UNSET, category=_UNSET, payload=_UNSET)

   Mark the beginning of a (nested, per-thread) code range.

   Args:
       message:
           A message associated with the annotated code range.

       color:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       domain:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       category:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       payload:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

   Note:
       When applicable, prefer `~.annotate`.


.. py:function:: pop_range(domain=_UNSET)

   Mark the end of a code range started with `~.push_range`.

   Args:
       domain:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).


.. py:function:: start_range(message=None, color=_UNSET, domain=_UNSET, category=_UNSET, payload=_UNSET)

   Mark the beginning of a process range.

   Args:
       message:
           A message associated with the range.

       color:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       domain:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       category:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

       payload:
           Accepted for compatibility; **dropped** (supplying it triggers the
           compatibility mode).

   Returns:
       A ``(range_id, domain_handle)`` tuple that must be passed to
       `~.end_range`. ``domain_handle`` is always ``0`` in this shim.


.. py:function:: end_range(range_id)

   Mark the end of a process range started with `~.start_range`.

   Args:
       range_id:
           The tuple (or bare id) returned by `~.start_range`.


.. py:class:: Profile(linenos=True, annotate_cfuncs=True)

   Programmatically control NVTX automatic annotations.

   Wraps every Python (and, optionally, C-extension) function call in a ROCTX
   range using ``sys.setprofile`` / ``threading.setprofile``.

   Args:
       linenos:
           Include file and line number information in annotations.

       annotate_cfuncs:
           Also annotate C-extension and builtin functions.

   Examples:
       >>> import nvtx, time
       >>> pr = nvtx.Profile()
       >>> pr.enable()
       >>> time.sleep(1)  # captured by nvtx
       >>> pr.disable()
       >>> time.sleep(1)  # not captured


   .. py:attribute:: linenos
      :value: True



   .. py:attribute:: annotate_cfuncs
      :value: True



   .. py:method:: enable()

      Start annotating function calls automatically.



   .. py:method:: disable()

      Stop annotating function calls automatically.



.. py:class:: RegisteredString(domain=None, string=None)

   A stand-in for an NVTX registered string.

   ROCTX has no string-registration API, so this simply wraps the raw string;
   the message is re-sent as a plain string on every event.


   .. py:attribute:: domain
      :value: None



   .. py:attribute:: string
      :value: None



.. py:class:: EventAttributes(domain=None, message=None, color=None, category=None, payload=None)

   A holder for event attributes.

   Only ``message`` reaches ROCTX; ``color``, ``category`` and ``payload``
   are stored for compatibility but **dropped**.


   .. py:attribute:: domain
      :value: None



   .. py:attribute:: message
      :value: None



   .. py:attribute:: color
      :value: None



   .. py:attribute:: category
      :value: None



   .. py:attribute:: payload
      :value: None



.. py:class:: Domain(name=None)

   A per-domain interface to the NVTX API.

   .. important::

      ROCTX has no domain concept. The ``name`` is accepted but ignored, and
      every method routes to the same global ROCTX calls, so events from
      different `~.Domain` instances are **not** isolated from one another.


   .. py:attribute:: name
      :value: None



   .. py:attribute:: handle
      :value: 0



   .. py:method:: push_range(attributes=None, **kwargs)

      Mark the beginning of a code range (see `~nvtx.push_range`).



   .. py:method:: pop_range()

      Mark the end of a code range (see `~nvtx.pop_range`).



   .. py:method:: mark(attributes=None, **kwargs)

      Mark an instantaneous event (see `~nvtx.mark`).



   .. py:method:: start_range(attributes=None, **kwargs)

      Mark the beginning of a process range (see `~nvtx.start_range`).



   .. py:method:: end_range(range_id)

      Mark the end of a process range (see `~nvtx.end_range`).



   .. py:method:: get_category_id(name)

      Return a synthetic category id (not honored by ROCTX).



   .. py:method:: get_registered_string(string)

      Return a `~.RegisteredString` wrapper (no C registration).



   .. py:method:: get_event_attributes(message=None, color=None, category=None, payload=None)

      Create an `~.EventAttributes` object.



   .. py:method:: set_event_attributes(attributes, *, message=_DONT_SET, color=_DONT_SET, category=_DONT_SET, payload=_DONT_SET)

      Set attributes on an existing `~.EventAttributes` object.



   .. py:method:: get_timestamp()

      Return a monotonic timestamp (for API compatibility only).



   .. py:method:: get_counter(name, dtype, *, description=None, scope=None, semantics=None, time_domain=None)

      Return a no-op counter (ROCTX has no counter API).



.. py:class:: DummyDomain

   A no-op replacement for `~.Domain` used when NVTX is disabled.


   .. py:attribute:: handle
      :value: 0



   .. py:attribute:: name
      :value: None



   .. py:method:: push_range(attributes=None, **kwargs)


   .. py:method:: pop_range()


   .. py:method:: mark(attributes=None, **kwargs)


   .. py:method:: start_range(attributes=None, **kwargs)


   .. py:method:: end_range(range_id)


   .. py:method:: get_category_id(name)


   .. py:method:: get_registered_string(string)


   .. py:method:: get_event_attributes(message=None, color=None, category=None, payload=None)


   .. py:method:: set_event_attributes(attributes, **kwargs)


   .. py:method:: get_timestamp()


   .. py:method:: get_counter(name, dtype, **kwargs)


.. py:function:: get_domain(name=None)

   Get or create a `~.Domain` for a domain name.

   Returns a `~.DummyDomain` when NVTX is disabled.


.. py:class:: Counter(domain=None, name=None, dtype=None)

   Base class for NVTX counters.

   .. important::

      ROCTX has no counter API, so all counter operations in this shim are
      **no-ops** that record nothing.


   .. py:attribute:: domain
      :value: None



   .. py:attribute:: name
      :value: None



   .. py:attribute:: dtype
      :value: None



   .. py:method:: sample(value)

      Record one counter sample (no-op).



   .. py:method:: sample_no_value(reason)

      Record a missing counter sample (no-op).



   .. py:method:: batch_submit(values, timestamps)

      Record a batch of counter samples (no-op).



.. py:class:: Int64Counter(domain=None, name=None)

   Bases: :py:obj:`Counter`


   Counter for signed 64-bit integer samples (no-op).


.. py:class:: Float64Counter(domain=None, name=None)

   Bases: :py:obj:`Counter`


   Counter for double-precision floating-point samples (no-op).


.. py:class:: ExtCounter(domain=None, name=None, dtype=None)

   Bases: :py:obj:`Counter`


   Counter for NumPy dtype-based samples and counter groups (no-op).


.. py:class:: DummyCounter(domain=None, name=None, dtype=None)

   Bases: :py:obj:`Counter`


   A no-op replacement for `~.Counter` when the domain is disabled.


.. py:class:: CounterSemantics(unit=None, value_type=None, interpolation=None, min=None, max=None, unit_scale_numerator=1, unit_scale_denominator=1)

   Metadata describing how a counter value should be interpreted.

   Stored for API compatibility only; it has no runtime effect on ROCTX.


   .. py:attribute:: unit
      :value: None



   .. py:attribute:: value_type


   .. py:attribute:: interpolation


   .. py:attribute:: min
      :value: None



   .. py:attribute:: max
      :value: None



   .. py:attribute:: unit_scale_numerator
      :value: 1



   .. py:attribute:: unit_scale_denominator
      :value: 1



.. py:class:: CounterValueType

   Bases: :py:obj:`enum.IntEnum`


   How counter sample values relate to previous samples.


   .. py:attribute:: ABSOLUTE
      :value: 0



   .. py:attribute:: DELTA
      :value: 1



   .. py:attribute:: DELTA_SINCE_START
      :value: 2



.. py:class:: CounterInterpolation

   Bases: :py:obj:`enum.IntEnum`


   How tools should interpolate counter values between samples.


   .. py:attribute:: POINT
      :value: 0



   .. py:attribute:: SINCE_LAST
      :value: 1



   .. py:attribute:: UNTIL_NEXT
      :value: 2



   .. py:attribute:: LINEAR
      :value: 3



.. py:class:: CounterNoValueReason

   Bases: :py:obj:`enum.IntEnum`


   Reasons for recording a counter sample without a value.


   .. py:attribute:: ZERO
      :value: 0



   .. py:attribute:: UNCHANGED
      :value: 1



   .. py:attribute:: UNAVAILABLE
      :value: 2



.. py:class:: TimestampType

   Bases: :py:obj:`enum.IntEnum`


   Timestamp domains that can be associated with batched samples.


   .. py:attribute:: NONE
      :value: 0



   .. py:attribute:: TOOL_PROVIDED
      :value: 1



   .. py:attribute:: CPU_TSC
      :value: 2



   .. py:attribute:: CPU_CLOCK_GETTIME_REALTIME
      :value: 3



   .. py:attribute:: CPU_CLOCK_GETTIME_MONOTONIC
      :value: 4



   .. py:attribute:: GPU_GLOBALTIMER
      :value: 5



.. py:function:: numpy_dtype(*args, counter_semantics=None, **kwargs)

   Construct a NumPy dtype, optionally carrying counter semantics metadata.

   Accepts the same arguments as ``numpy.dtype``. The ``counter_semantics``
   metadata is attached but has **no runtime effect** on ROCTX.

   Raises:
       ``RuntimeError``:
           If NumPy is not installed.


