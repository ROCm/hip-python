# hip-python
HIP Python Low-level Bindings 

## Build requirements

> * Commit used for evaluation : 9a5505be427b95ead61a45c0f6e89c1ba2edc0ef
> * Date: 04/21/2023

Python >= 3.7 is required.

The following pip packages are required for running setup.py 

* ``cython``
* ``setuptools`` >= 10.0.1

* ``libclang`` (>= 14.0.1, != 15.0.3, != 15.0.6, <= 15.0.6.1)

  The following ``libclang`` versions have been tested:

  * 9.0.1	-- fail
  * 10.0.1.0 -- fail
  * 11.0.0 -- fail
  * 11.0.1 -- fail
  * 11.1.0 -- fail
  * 12.0.0 -- fail
  * 13.0.0 -- fail
  * 14.0.1 -- **success**
  * 14.0.6 -- **success**
  * 15.0.3 -- fail
  * 15.0.6 -- fail
  * 15.0.6.1 -- **success**
  * 16.0.0 -- fail

#### Legal Requirements:
Always include the appropriate copyright and MIT X11 notice (see below)
* at the top of the AMD developed files, and
* in a LICENSE text file in the top level directory.

#### Standard ongoing code readiness and release obligations:
* Follow the Developer Guidelines here: http://confluence.amd.com/pages/viewpage.action?pageId=52793191

```
MIT License
 
Copyright (c) <year> Advanced Micro Devices, Inc.
 
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
