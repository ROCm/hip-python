rocm.bindings.roctx
===================

.. py:module:: rocm.bindings.roctx


Attributes
----------

.. autoapisummary::

   rocm.bindings.roctx.ROCTX_VERSION_MAJOR
   rocm.bindings.roctx.ROCTX_VERSION_MINOR


Functions
---------

.. autoapisummary::

   rocm.bindings.roctx.has_symbol
   rocm.bindings.roctx.roctx_version_major
   rocm.bindings.roctx.roctx_version_minor
   rocm.bindings.roctx.roctxMarkA
   rocm.bindings.roctx.roctxRangePushA
   rocm.bindings.roctx.roctxRangePop
   rocm.bindings.roctx.roctxRangeStartA
   rocm.bindings.roctx.roctxRangeStop


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: ROCTX_VERSION_MAJOR
   :type:  Any

.. py:data:: ROCTX_VERSION_MINOR
   :type:  Any

.. py:function:: roctx_version_major()

   Query the major version of the installed library.

   Return the major version of the installed library. This can be used to check
   if it is compatible with this interface version.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: Returns the major version number.

   .. rubric:: C signature

   .. code-block:: c

       uint32_t roctx_version_major()


.. py:function:: roctx_version_minor()

   Query the minor version of the installed library.

   Return the minor version of the installed library. This can be used to check
   if it is compatible with this interface version.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: Returns the minor version number.

   .. rubric:: C signature

   .. code-block:: c

       uint32_t roctx_version_minor()


.. py:function:: roctxMarkA(message)

   Mark an event.

   Args:
       message (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The message associated with the event.

   .. rubric:: C signature

   .. code-block:: c

       void roctxMarkA(const char * message)


.. py:function:: roctxRangePushA(message)

   Start a new nested range.

   Nested ranges are stacked and local to the current CPU thread.

   Args:
       message (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The message associated with this range.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: Returns the level this nested range is started at. Nested range
           levels are 0 based.

   .. rubric:: C signature

   .. code-block:: c

       int roctxRangePushA(const char * message)


.. py:function:: roctxRangePop()

   Stop the current nested range.

   Stop the current nested range, and pop it from the stack. If a nested range
   was active before the last one was started, it becomes again the current
   nested range.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: Returns the level the stopped nested range was started at, or a
           negative value if there was no nested range active.

   .. rubric:: C signature

   .. code-block:: c

       int roctxRangePop()


.. py:function:: roctxRangeStartA(message)

   Starts a process range.

   Start/stop ranges can be started and stopped in different threads. Each
   timespan is assigned a unique range ID.

   Args:
       message (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`) -- *IN*:
           The message associated with this range.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: Returns the ID of the new range.

   .. rubric:: C signature

   .. code-block:: c

       roctx_range_id_t roctxRangeStartA(const char * message)


.. py:function:: roctxRangeStop(id)

   Stop a process range.

   Args:
       id (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void roctxRangeStop(roctx_range_id_t id)


