cuda.core
=========

.. py:module:: cuda.core

.. autoapi-nested-parse::

   Minimal ``cuda.core`` compatibility shim, backed by HIP.

   This is NOT a full port of NVIDIA's ``cuda.core`` / ``cuda.core.experimental``
   package. It implements just enough of the high-level surface for HIP ports of
   CUDA-Python consumers: a `~.Device` exposing ``uuid``, the `~.Stream`
   vocabulary type behind the CUDA stream protocol, and stream-ordered device
   allocation through `~.DeviceMemoryResource` and `~.Buffer`.

   Everything is implemented on top of the high-level ``rocm.bindings.hip`` HIP
   runtime API. Additional high-level abstractions (Event, Program, Linker, ...)
   are intentionally out of scope and can be added on demand.



Classes
-------

.. autoapisummary::

   cuda.core.Device
   cuda.core.Stream
   cuda.core.Buffer
   cuda.core.DeviceMemoryResource


Package Contents
----------------

.. py:class:: Device(device_id=None)

   HIP-backed stand-in for ``cuda.core.Device``.

   Only the members required by current consumers are implemented. Passing no
   ``device_id`` selects the current device (via ``hipGetDevice``).


   .. py:property:: device_id
      :type: int


      Ordinal of the underlying HIP device.



   .. py:property:: uuid
      :type: str


      Device UUID as a ``GPU-<hex>`` string.

      Backed by ``hipDeviceGetUuid``. ROCm stores the UUID as ASCII text in
      the 16-byte field (e.g. ``b"50a437de77a657b8"``), so the result matches
      the ``GPU-<hex>`` string reported by ``rocminfo`` / ``amd-smi``. Falls
      back to a raw hex rendering if a runtime ever returns non-text bytes.



   .. py:method:: set_current()


   .. py:method:: sync()


   .. py:property:: default_stream

      The NULL stream, as a non-owning `~.Stream` token.

      HIP has no per-thread default stream selected by an environment
      variable the way CUDA Python's
      ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` does, so this is always
      the legacy NULL stream. The token is created once per `~.Device` so
      that repeated reads compare equal.



   .. py:method:: create_stream(obj=None)

      Create a HIP stream, or wrap a foreign one.

      With no argument a new non-blocking stream is created --- non-blocking
      being the ``cuda.core`` default --- and destroyed again when the
      returned `~.Stream` is closed or collected. HIP associates a new stream
      with the *current* device rather than with the device this method was
      called on, so the two must agree.

      Passing ``obj`` instead wraps an existing stream from any library
      implementing the CUDA stream protocol. The wrapper keeps ``obj`` alive
      and never destroys the borrowed stream.



.. py:class:: Stream(*args, **kwargs)

   HIP-backed stand-in for ``cuda.core.Stream``.

   Instances come from :py:meth:`Device.create_stream` or
   :py:attr:`Device.default_stream`. As in ``cuda.core``, constructing one
   directly is refused: whether the stream would be owned or borrowed is
   ambiguous, and the two differ in what closing them does.


   .. py:property:: handle

      The underlying ``hipStream_t``.

      As in ``cuda.core``, this is a Python object; use ``int()`` on it to
      get the address of the C handle.



   .. py:method:: sync()

      Block until all work queued on this stream has completed.



   .. py:method:: close()

      Destroy an owned stream, or drop the reference to a borrowed one.



.. py:class:: Buffer(*args, **kwargs)

   HIP-backed stand-in for ``cuda.core.Buffer``.

   Instances come from :py:meth:`DeviceMemoryResource.allocate`, and hold on
   to the stream that allocated them: closing without naming a stream frees
   the allocation in the order it was created, and a stream-ordered free onto
   an already destroyed stream would not be valid.


   .. py:property:: handle

      The allocation, as the ``DeviceArray`` HIP returned for it.

      As in ``cuda.core``, this is a Python object; use ``int()`` on it to
      get the device address.



   .. py:property:: size

      Size of the allocation in bytes.



   .. py:method:: close(stream=None)

      Free the allocation, ordered on ``stream``.

      ``stream`` defaults to the one that allocated the buffer. The free is
      stream-ordered, so it takes effect once preceding work on that stream
      has run, not when this call returns.



.. py:class:: DeviceMemoryResource(device_id)

   HIP-backed stand-in for ``cuda.core.DeviceMemoryResource``.

   Allocations are stream-ordered and come from the device's current HIP
   memory pool, which is the device's default pool unless one has been set
   with ``hipDeviceSetMemPool``. The pool therefore outlives this object and
   is never destroyed by it, matching how ``cuda.core`` treats a memory
   resource built over a pool it does not own.


   .. py:property:: device_id

      Ordinal of the device whose pool backs this resource.



   .. py:property:: handle

      The underlying ``hipMemPool_t``.



   .. py:method:: allocate(size, stream=None)

      Allocate ``size`` bytes from the pool, ordered on ``stream``.

      The allocation is only safe to access once preceding work on
      ``stream`` has run. ``stream`` defaults to the NULL stream.



   .. py:method:: deallocate(ptr, size, stream=None)

      Return ``ptr`` to the pool, ordered on ``stream``.



