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

rocm.bindings.util
==================

The ``rocm.bindings.util`` package is the foundation of the modern hip-python
package layout. It is **handcoded in full** — no part of it is produced by
the interfacegen code generator. Every other ``rocm-bindings-*`` package
depends on it for DLL loading, type marshalling, and ROCm path resolution.

It contains:

* The platform-agnostic :py:mod:`rocm.bindings.util.loader` Cython module
  (dispatching to ``posixloader`` on Linux or ``win32loader`` on Windows at
  Cython compile time).
* The shared Cython type adapters in :py:mod:`rocm.bindings.util.types`
  (``Pointer``, ``CStr``, ``ImmortalCStr``, ``NDBuffer``, ``ListOfPointer``,
  ``ListOfBytes``, ``DeviceArray``).
* The :py:mod:`rocm.bindings.util.paths` helper with
  ``get_library_path('<libname>')`` for lazy ROCm library resolution at
  import time.

.. autoapi-module:: rocm.bindings.util
