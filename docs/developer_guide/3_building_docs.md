<!-- MIT License
  -- 
  -- Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->
# Build the Docs

TBC

* Building the documentation requires to build the packages first

* Sphinx does not perform a syntactical
  analysis but instead loads the
  packages and reads out the docstrings
  from each python object.

* in order to have access to the 

## On the used MyST syntax

For more details, see

* [MyST Syntax](https://myst-parser.readthedocs.io/en/v0.16.1/syntax/syntax.html)
* [How to use cross-references with Sphinx](https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html)

## Tips and Tricks

### Speed Up Docs Development Build if API is not Important

In scenarios where you just want to work on the manually written documentation
files, you might not want to build the API documentation every time, which
takes a considerable time.

In this case, you can move the ``hip-python/docs/python_api`` subfolder to a higher level in the file tree, e.g. in ``hip-python``.
Sphinx then won't find and parse the files in this folder anymore. 

:::{note}

This trick obviously will result in dead links within all other documents that refer to the API documentation pages. 
:::

:::{note}

The next call to the code generator will recreate the ``hip-python/docs/python_api`` folder.
You can either rely on this behavior or restore the folder manually before building the final docs.
:::

## Famous last words

Feel encouraged to run a spell checker tool such as [``aspell`](http://aspell.net/) on the
documentation text files that you have modified before comitting them to the repository.