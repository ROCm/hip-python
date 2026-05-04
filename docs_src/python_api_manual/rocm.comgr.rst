.. MIT License
..
.. Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

rocm.comgr
==========

The ``rocm.comgr`` package is a **handcoded** high-level Python wrapper around
the generator-emitted COMGR Cython bindings (``rocm.bindings.amd_comgr``,
``rocm.bindings.cyamd_comgr``). It ships in the ``rocm-bindings-compiler``
wheel under ``python/rocm-bindings-compiler/rocm/comgr/``.

It provides convenience helpers for working with AMD Code Object Manager
(COMGR) workflows: assembling and disassembling code objects, parsing AMD HSA
kernel descriptors, etc.

.. seealso::

   * :doc:`/python_api/rocm.bindings.amd_comgr` — the generator-emitted
     low-level Python bindings.

.. autoapi-module:: rocm.comgr

Submodules
----------

.. autoapi-module:: rocm.comgr.comgr

.. autoapi-module:: rocm.comgr.amd_hsa_kernel_descriptor

.. autoapi-module:: rocm.comgr.amdhsa_kernel_directives
