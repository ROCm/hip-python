rocm.hipfile.buffer
===================

.. py:module:: rocm.hipfile.buffer


Classes
-------

.. autoapisummary::

   rocm.hipfile.buffer.Buffer


Module Contents
---------------

.. py:class:: Buffer(buffer_ptr, length, flags)

   Lifecycle manager for a hipFile-registered GPU memory region.

   The caller pre-allocates the underlying device buffer (typically
   via `~.rocm.bindings.hip.hipMalloc` or any equivalent
   GPU-allocator) and constructs a `~.Buffer` to register it
   with the hipFile driver. Deregistration happens on context exit
   (or via explicit `~.deregister`).

   Use as a context manager:

       with Buffer(ptr, length, flags=0) as registered_buf:
           ...


   .. py:method:: from_ctypes_void_p(ctypes_void_p: ctypes.c_void_p, length, flags)
      :classmethod:


      Construct a `~.Buffer` from a :py:class:`ctypes.c_void_p`.



   .. py:property:: ptr


   .. py:method:: deregister()


   .. py:method:: register()


