rocm.hipfile.properties
=======================

.. py:module:: rocm.hipfile.properties


Functions
---------

.. autoapisummary::

   rocm.hipfile.properties.driver_get_properties
   rocm.hipfile.properties.get_version


Module Contents
---------------

.. py:function:: driver_get_properties()

   Return the driver properties struct as set by `~.hipFileDriverGetProperties`.


.. py:function:: get_version()

   Return ``(major, minor, patch)`` for the loaded libhipfile.so.


