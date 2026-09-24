hip
===

.. py:module:: hip

.. autoapi-nested-parse::

   hip-python - Backward compatibility package for ROCm Python bindings.

   This package provides backward compatibility with the old hip-python package structure.
   It re-exports modules from the new rocm.bindings namespace.

   New code should use: from rocm.bindings import hip, hiprtc, hipblas, etc.
   Old code continues to work: from hip import hip, hiprtc, hipblas, etc.

   `hip`, `hiprtc` and `hip._util` come with this wheel's hard dependencies.
   Every other binding is imported below if its wheel is installed, and its name
   stays unbound otherwise.



