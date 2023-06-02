# HIP Python

Low-level Python and Cython Bindings for HIP.

## Build requirements

Python >= 3.7 is required plus Python development files (e.g. via ``apt install python3-dev`` on Ubuntu).

To build ``pip`` packages (``.whl``) you need to install the ``pip`` package ``build``.
You further need to have `venv` installed (e.g. via ``apt install python3-venv`` on Ubuntu).

The following ``pip`` packages are required for running ``setup.py``

* ``cython``
* ``setuptools`` >= 10.0.1

#### Legal Requirements:

Always include the appropriate copyright and MIT X11 notice (see below)
* at the top of the AMD developed files, and
* in a LICENSE text file in the top level directory.

#### Standard ongoing code readiness and release obligations:

* Follow the Developer Guidelines here: http://confluence.amd.com/pages/viewpage.action?pageId=52793191

```
MIT License
 
Copyright (c) 2023 Advanced Micro Devices, Inc.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
