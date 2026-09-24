rocm.version
============

.. py:module:: rocm.version

.. autoapi-nested-parse::

   Version information for ROCm bindings.

   The ROCm/HIP versions and code-generation provenance are baked in as plain
   constants at codegen time. This module performs no package-metadata or TOML
   parsing to discover the ROCm/HIP version.

   Usage:
       from rocm import version

       print(version.ROCM_VERSION_TUPLE)  # (7, 13, 0)
       print(version.ROCM_VERSION_NAME)   # "7.13.0"
       print(version.ROCM_VERSION)        # 71300000
       print(version.HIP_VERSION_TUPLE)   # (7, 13, 26154, "92b7431876")

       # Or import specific attributes:
       from rocm.version import ROCM_VERSION_TUPLE, HIP_VERSION_TUPLE



Attributes
----------

.. autoapisummary::

   rocm.version.HIP_COMMIT
   rocm.version.HIP_FULL_VERSION
   rocm.version.BASE_BRANCH
   rocm.version.BASE_REV
   rocm.version.BASE_VERSION
   rocm.version.INTERFACEGEN_VERSION
   rocm.version.ROCM_VERSION
   rocm.version.HIP_VERSION


Module Contents
---------------

.. py:data:: HIP_COMMIT
   :value: '0000000'


.. py:data:: HIP_FULL_VERSION
   :value: '7.15.26333-0000000'


.. py:data:: BASE_BRANCH
   :value: 'amd-integration'


.. py:data:: BASE_REV
   :value: '183b6a80'


.. py:data:: BASE_VERSION
   :value: '0.1.2'


.. py:data:: INTERFACEGEN_VERSION
   :value: '0.5'


.. py:data:: ROCM_VERSION

.. py:data:: HIP_VERSION

