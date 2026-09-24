rocm.hipfile.error
==================

.. py:module:: rocm.hipfile.error


Exceptions
----------

.. autoapisummary::

   rocm.hipfile.error.HipFileException


Module Contents
---------------

.. py:exception:: HipFileException(hipfile_err, hip_err)

   Bases: :py:obj:`Exception`


   Exception raised on a non-success hipFile error.

   Carries both the hipFile-level error code (``hipfile_err``) and the
   underlying HIP driver error (``hip_err``) when the former is
   ``OpError.HIP_DRIVER_ERROR``.


   .. py:property:: hipfile_err


   .. py:property:: hip_err


