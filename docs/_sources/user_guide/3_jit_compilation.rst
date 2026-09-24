.. MIT License
..
.. Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

.. _ch_jit_compilation:

JIT Compilation and LLVM IR
===========================

This chapter shows how to compile, link, and inspect HIP kernels at
runtime, and how to work on the LLVM IR that sits between HIP source
and a finished code object. The basic ``hiprtc`` flow --- "compile a
HIP source string, load the resulting code object as a module,
launch a kernel" --- is covered in :ref:`sec_launching_kernels`.
Here we go beyond that:

- linking multiple translation units (separable compilation),
- linking HIP code with externally supplied LLVM IR,
- using **AMD COMGR** as a lower-level alternative to HIPRTC for
  pipeline stages that HIPRTC does not expose (e.g. emitting HSA
  assembly, parsing code-object metadata, disassembling kernels),
- parsing, building, transforming and running LLVM IR through the
  LLVM-C bindings.

Three Python-level entry points are used throughout:

- :py:obj:`~.hiprtc` --- the high-level "HIP source --> GPU
  executable" runtime compiler. Lives in
  ``rocm.bindings.hiprtc``. Best when you just want a kernel to
  run.
- :py:obj:`rocm.comgr` --- a handcoded high-level wrapper over the
  raw :py:obj:`~.amd_comgr` C bindings (in
  ``rocm.bindings.amd_comgr``). Exposes individual pipeline
  stages (``compile_hip_to_bc``, ``compile_bc_to_hsa``,
  ``parse_code_obj_metadata``, ``disassemble_code_obj_function``,
  ...). Use it when you need to produce or inspect intermediate
  artefacts.
- ``rocm.bindings.llvm.c`` --- bindings for the LLVM-C API of the
  LLVM that comes with ROCm. Use them when the artefact you care
  about is **the IR itself** rather than a kernel that runs.

The first two take HIP source in and give GPU code objects back,
and they know about AMDGPU targets, runtime headers and code-object
metadata. The third knows nothing about GPUs: it is the same LLVM-C
API that upstream documents, operating on modules in memory. The
first half of this chapter covers the two HIP compiler APIs, the
second half the LLVM bindings; :ref:`sec_llvm_handoff` describes how
bitcode passes between them.

.. note::

   The example files referenced below define a ``hip_check`` helper
   that raises :py:obj:`RuntimeError` on a non-success return
   status (and an analogous ``llvm_check``). The page snippets
   below omit those helpers --- if you copy a snippet, wrap each
   ``hip.``/``hiprtc.`` call in your own status check.

Compiling HIP at runtime with HIPRTC and AMD COMGR
--------------------------------------------------

.. _sec_jit_hiprtc_linking:

Linking HIP translation units with HIPRTC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to compile a HIP source to **LLVM bitcode** (instead of a
     finished code object) with the ``-fgpu-rdc`` flag.
   * How to link multiple bitcode modules into one code object
     using :py:obj:`~.hiprtcLinkCreate`,
     :py:obj:`~.hiprtcLinkAddData`, and
     :py:obj:`~.hiprtcLinkComplete`.

When a HIP source contains a call to a ``__device__`` function
defined in a *different* translation unit, the compiler can't
resolve it at compile time. The standard fix is **separable
compilation**: each unit is compiled to LLVM bitcode (option
``-fgpu-rdc``, "relocatable device code"), and a runtime linker
merges the bitcode modules into one code object before loading.

The two HIP sources --- a kernel that calls an unresolved device
function ``foo()``, and a translation unit that defines ``foo()``:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_linking_device_functions.py
   :language: python
   :start-after: [literalinclude-kernel-sources-begin]
   :end-before: [literalinclude-kernel-sources-end]
   :dedent:
   :name: jit_hiprtc_linking_kernel_sources
   :caption: HIP sources --- kernel and unresolved device function

Each source is compiled with ``-fgpu-rdc``; instead of pulling
out the finished code object via :py:obj:`~.hiprtcGetCode`, we
ask for the LLVM bitcode via :py:obj:`~.hiprtcGetBitcode`:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_linking_device_functions.py
   :language: python
   :start-after: [literalinclude-hiprtc-compile-rdc-begin]
   :end-before: [literalinclude-hiprtc-compile-rdc-end]
   :dedent:
   :name: jit_hiprtc_linking_compile
   :caption: Compile each HIP source to LLVM bitcode with ``-fgpu-rdc``

Both bitcode modules are then fed to a HIPRTC link state, the
linker is asked to complete, and the resulting code object is
loaded as an ordinary :py:obj:`~.hipModule_t`:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_linking_device_functions.py
   :language: python
   :start-after: [literalinclude-hiprtc-link-flow-begin]
   :end-before: [literalinclude-hiprtc-link-flow-end]
   :dedent:
   :name: jit_hiprtc_linking_link_flow
   :caption: Link two bitcode modules and load the resulting code object

.. admonition:: What is happening?

   1. :py:obj:`~.hiprtcLinkCreate` returns a fresh link state.
   2. Each program's bitcode buffer is passed to
      :py:obj:`~.hiprtcLinkAddData` with the input type
      :py:obj:`~.hiprtcJITInputType.HIPRTC_JIT_INPUT_LLVM_BITCODE`
      --- the linker consumes LLVM bitcode, *not* native code
      objects.
   3. :py:obj:`~.hiprtcLinkComplete` produces the final code-object
      bytes, which load just like any other module via
      :py:obj:`~.hipModuleLoadData` /
      :py:obj:`~.hipModuleGetFunction`.

   See ``examples/2_Advanced/hiprtc_linking_device_functions.py``
   for the full runnable program.

.. _sec_jit_hiprtc_llvm_ir:

Linking HIP with externally supplied LLVM IR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * That :py:obj:`~.hiprtcLinkAddData` accepts hand-written
     LLVM IR as long as it is presented as bitcode.
   * The :py:obj:`~.hipJitInputType` /
     :py:obj:`~.hiprtcJITInputType` enum compatibility shim for
     the ``LLVM_BITCODE`` input.

The previous section linked two HIPRTC-produced bitcode modules.
The same link API also accepts LLVM IR you produced by other
means --- for example, by ``clang``-compiling a different
language to AMDGPU IR --- as long as you present it as bitcode.
The example file ``hiprtc_linking_with_llvm_ir.py`` demonstrates
this with three inputs: two HIP translation units (compiled with
``-fgpu-rdc`` as in the previous section) plus a hand-supplied
LLVM IR module that defines a missing ``__device__`` function. The
IR-to-bitcode conversion in that example goes through the LLVM-C
bindings of the second half of this chapter; see
:ref:`sec_llvm_handoff`.

.. note::

   The hand-supplied LLVM IR in the example file is
   target-dependent (built for ``gfx90a``); see the example file
   for the literal IR string. The example will exit with an
   error on non-``gfx90a`` GPUs.

The input-type enum moved between ROCm releases. The example
prefers the new :py:obj:`~.hipJitInputType.hipJitInputLLVMBitcode`
(>= ROCm 6.4) and falls back to the legacy
:py:obj:`~.hiprtcJITInputType.HIPRTC_JIT_INPUT_LLVM_BITCODE`:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_linking_with_llvm_ir.py
   :language: python
   :start-after: [literalinclude-hiprtc-link-input-type-begin]
   :end-before: [literalinclude-hiprtc-link-input-type-end]
   :dedent:
   :name: jit_hiprtc_llvm_ir_input_type
   :caption: ``LLVM_BITCODE`` input-type compatibility shim

The link orchestration mixes bitcode from HIPRTC-compiled sources
(``kernel_prog``, ``print_val_prog``) with bitcode produced from
the hand-supplied LLVM IR (``scale_op_prog``):

.. literalinclude:: ../../examples/2_Advanced/hiprtc_linking_with_llvm_ir.py
   :language: python
   :start-after: [literalinclude-hiprtc-link-mixed-begin]
   :end-before: [literalinclude-hiprtc-link-mixed-end]
   :dedent:
   :name: jit_hiprtc_llvm_ir_link
   :caption: Link mixed HIP-bitcode and LLVM-IR-bitcode inputs

.. _sec_jit_comgr_hip_to_bc:

HIP source --> LLVM bitcode via AMD COMGR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to drive the HIP --> LLVM-bitcode stage of the AMD
     compiler pipeline directly via :py:obj:`rocm.comgr`, without
     going through HIPRTC.
   * Why HIP source compiled by COMGR must be prefixed with
     :py:obj:`rocm.comgr.HIPRTC_RUNTIME_HEADER`.

HIPRTC is convenient but treats the compilation pipeline as a
black box. AMD COMGR exposes individual stages (preprocess,
compile-to-bitcode, link-bitcode, compile-bitcode-to-relocatable,
assemble, link-executable) as standalone calls. The
:py:obj:`rocm.comgr` module wraps the most common end-to-end
recipes as one-call helpers; the first useful one is
:py:obj:`~rocm.comgr.compile_hip_to_bc`.

Unlike HIPRTC, COMGR's HIP frontend does *not* automatically
prepend ``hip/hip_runtime.h`` to your source. The
:py:obj:`rocm.comgr.HIPRTC_RUNTIME_HEADER` constant exposes the
exact header that HIPRTC uses internally so you can prepend it
yourself. It is read on first access from the ``hiprtc-builtins``
library of the ROCm installation in use, so the text always
matches that ROCm and the platform you are compiling on; see
:py:mod:`rocm.comgr.hiprtc_header` if you need to override where
it comes from.

.. literalinclude:: ../../examples/2_Advanced/amd_comgr_hip_to_llvm_ir.py
   :language: python
   :start-after: [literalinclude-comgr-runtime-header-begin]
   :end-before: [literalinclude-comgr-runtime-header-end]
   :dedent:
   :name: jit_comgr_hip_runtime_header
   :caption: Prepend ``HIPRTC_RUNTIME_HEADER`` to the HIP source

The compile call itself returns the bitcode bytes, the COMGR
log, and a structured diagnostic:

.. literalinclude:: ../../examples/2_Advanced/amd_comgr_hip_to_llvm_ir.py
   :language: python
   :start-after: [literalinclude-comgr-compile-hip-to-bc-begin]
   :end-before: [literalinclude-comgr-compile-hip-to-bc-end]
   :dedent:
   :name: jit_comgr_compile_hip_to_bc
   :caption: ``comgr.compile_hip_to_bc`` --- HIP source --> LLVM bitcode

The ``isa_name`` follows the standard AMDGPU triple
(``amdgcn-amd-amdhsa--<arch>``). The ``hip_version_tuple`` is
needed because the runtime header pre-declares macros that depend
on the active ROCm release; the example file pulls it from
:py:obj:`rocm.version.ROCM_VERSION_TUPLE`.

.. _sec_jit_comgr_bc_to_hsa:

LLVM bitcode --> HSA assembly via AMD COMGR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to emit human-readable HSA assembly from LLVM bitcode
     via :py:obj:`~rocm.comgr.compile_bc_to_hsa`.
   * That this fills a gap in HIPRTC, which currently has no
     "stop after assembly" mode.

HIPRTC compiles HIP sources straight through to a relocatable
code object; there is no public API to halt the pipeline at the
assembly stage. COMGR exposes that stage as
:py:obj:`~rocm.comgr.compile_bc_to_hsa`:

.. literalinclude:: ../../examples/2_Advanced/amd_comgr_llvm_ir_to_hsa.py
   :language: python
   :start-after: [literalinclude-comgr-compile-bc-to-hsa-begin]
   :end-before: [literalinclude-comgr-compile-bc-to-hsa-end]
   :dedent:
   :name: jit_comgr_compile_bc_to_hsa
   :caption: ``comgr.compile_bc_to_hsa`` --- LLVM bitcode --> HSA assembly

The example file feeds in a hand-supplied LLVM IR module for
``gfx942``; that ~50-line IR string is omitted from this page.
You can regenerate the input from any HIP source by asking
``hipcc`` to halt after the LLVM IR stage:

.. code-block:: shell

   hipcc -emit-llvm -S --offload-arch=gfx942 vector_add.hip -o - \
       | sed -n "/hip-amdgcn-amd-amdhsa--gfx942/,/hip-amdgcn-amd-amdhsa--gfx942/p"

.. _sec_jit_comgr_inspect:

Inspecting a JIT-compiled code object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to read the metadata block embedded in a HIPRTC-produced
     code object.
   * How to enumerate kernel and data symbols in the object.
   * How to disassemble a single function from the code object.

A finished AMD GPU code object carries enough metadata to
reconstruct kernel signatures, kernel-descriptor layout, and
the assembly itself.
:py:obj:`rocm.comgr` wraps the four most common queries:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_amd_comgr_get_jit_kernel_metadata.py
   :language: python
   :start-after: [literalinclude-comgr-inspect-metadata-begin]
   :end-before: [literalinclude-comgr-inspect-metadata-end]
   :dedent:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_amd_comgr_get_jit_kernel_metadata.py
   :language: python
   :start-after: [literalinclude-comgr-inspect-kernel-names-begin]
   :end-before: [literalinclude-comgr-inspect-kernel-names-end]
   :dedent:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_amd_comgr_get_jit_kernel_metadata.py
   :language: python
   :start-after: [literalinclude-comgr-inspect-symbols-begin]
   :end-before: [literalinclude-comgr-inspect-symbols-end]
   :dedent:

.. literalinclude:: ../../examples/2_Advanced/hiprtc_amd_comgr_get_jit_kernel_metadata.py
   :language: python
   :start-after: [literalinclude-comgr-inspect-disassemble-begin]
   :end-before: [literalinclude-comgr-inspect-disassemble-end]
   :dedent:

A typical inspection loop calls them on a ``code`` /
``code_size`` pair returned by :py:obj:`~.hiprtcGetCode` (or by
:py:obj:`~.hiprtcLinkComplete`):

.. code-block:: python

   metadata     = comgr.parse_code_obj_metadata(code, code_size)
   kernel_names = comgr.parse_code_obj_kernel_names(code, code_size)
   symbols      = comgr.parse_code_symbols(code, code_size)
   for k in kernel_names:
       asm = comgr.disassemble_code_obj_function(
           code, f"amdgcn-amd-amdhsa--{arch}", func_name=k
       )

The structures returned by ``parse_code_obj_metadata`` and
``parse_code_symbols`` are plain Python dictionaries, suitable
for ``yaml.dump`` --- see the example file for a full walk-through
that prints metadata, symbols, and disassembly side by side.

For the documentation of ``rocm.comgr`` itself, see
:doc:`/python_api/rocm/comgr/index`.

.. _ch_llvm_bindings:

Working with LLVM IR through the LLVM-C bindings
------------------------------------------------

The ``rocm-bindings-compiler`` package also carries bindings for
the whole LLVM-C API of the LLVM that comes with ROCm, in the
``rocm.bindings.llvm.c`` package. They let a Python program do what
a C program linked against ``libLLVM`` can do: build a module in
memory, parse IR or bitcode, verify it, run pass pipelines over it,
query targets and data layouts, and hand the result to an execution
engine.

Nothing in this half is AMD GPU specific --- upstream's LLVM
documentation applies to every call named below. What the following
sections add is how that API looks after passing through HIP
Python's code generator.

.. note::

   ``numba.hip`` is the largest consumer of these bindings inside
   HIP Python: it builds, links and optimizes every kernel's IR
   through them before handing it to COMGR. See
   :doc:`/user_guide/4_numba_hip`.

.. _sec_llvm_modules:

How the modules are laid out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How an LLVM-C header name maps to a Python module name.
   * Which modules exist besides ``rocm.bindings.llvm.c.*``.
   * Where the Cython-level declarations live.

There is one Python module per LLVM-C header, and the mapping is
mechanical: the ``llvm-c`` directory becomes ``llvm.c``, the header
basename is lowercased, and subdirectories are kept.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - C header
     - Python module

   * - ``llvm-c/Core.h``
     - ``rocm.bindings.llvm.c.core``

   * - ``llvm-c/BitReader.h``
     - ``rocm.bindings.llvm.c.bitreader``

   * - ``llvm-c/ExecutionEngine.h``
     - ``rocm.bindings.llvm.c.executionengine``

   * - ``llvm-c/Transforms/PassBuilder.h``
     - ``rocm.bindings.llvm.c.transforms.passbuilder``

   * - ``llvm/Config/llvm-config.h``
     - ``rocm.bindings.llvm.config.llvm_config``

The set of modules follows whichever LLVM the bindings were
generated against, so it grows and shrinks with the ROCm release;
``analysis``, ``bitwriter``, ``debuginfo``, ``disassembler``,
``error``, ``irreader``, ``linker``, ``lljit``, ``object``, ``orc``,
``target``, ``targetmachine`` and ``types`` are among the modules
present in every supported release. Two of them are unusual:
``rocm.bindings.llvm.c.datatypes`` is generated from a header that
declares no functions and is therefore empty, and
``rocm.bindings.llvm.config.llvm_config`` exposes the macros of the
LLVM the wheel was **built** against, which need not describe the
library it loads at runtime.

Each module has a C-level companion for Cython users, named after
its leaf with a ``cy`` prefix --- ``rocm.bindings.llvm.c.cycore``
next to ``rocm.bindings.llvm.c.core``. Those declare the plain C
functions and types, so a ``cimport`` reaches ``libLLVM`` without
going through Python objects at all:

.. code-block:: cython

   from rocm.bindings.llvm.c.cycore cimport (
       LLVMModuleCreateWithName,
       LLVMPrintModuleToString,
   )

For the generated reference documentation of every module, see
:doc:`/python_api/rocm/bindings/llvm/index`.

.. _sec_llvm_conventions:

Calling conventions
^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * Why LLVM calls do not return a status tuple like the rest of
     HIP Python.
   * How strings, arrays and out-parameters cross the boundary.
   * Who owns the objects LLVM hands back.

**Return values.** Most HIP Python bindings return a status as the
first element of a tuple, because the libraries they wrap report
errors that way. LLVM-C has no such convention, so its wrappers
return what the C function returns:

.. code-block:: python

   mod   = LLVMModuleCreateWithName("my_module")   # a handle
   int32 = LLVMInt32Type()                         # a handle
   name  = LLVMGetTargetName(target)               # a CStr

A function with out-parameters appends them to the C return value,
which is where tuples come from. ``LLVMVerifyModule`` returns the
``LLVMBool`` plus its message, and
``LLVMCreateMemoryBufferWithContentsOfFile`` returns its status plus
the buffer and a message, so the caller checks the status itself:

.. literalinclude:: ../../examples/2_Advanced/parse_llvm_bitcode.py
   :language: python
   :start-after: [literalinclude-check-status-begin]
   :end-before: [literalinclude-check-status-end]
   :dedent:
   :name: llvm_check_status
   :caption: Checking a status that LLVM reports as a return value

**Strings.** Pass a plain :py:obj:`str` wherever the C API wants a
``const char *``; the adapter encodes it as UTF-8 and keeps the
buffer alive. Returned ``char *`` arrives as a
:py:obj:`~.types.CStr`, which decodes with
``str(...)`` and supports the buffer protocol --- see
:ref:`sec_cstr`.

**Arrays.** Parameters that take an array of handles, such as the
parameter types of ``LLVMFunctionType`` or the arguments of
``LLVMRunFunction``, accept an ordinary Python list. The adapter
that backs the list must outlive the call, which it does when the
list is passed inline as an argument.

**Ownership.** Nothing is reference counted. Every LLVM object the
bindings hand out is freed by the matching ``LLVMDispose*`` call,
and there are several of them: ``LLVMDisposeMessage`` for strings
LLVM allocated, ``LLVMDisposeErrorMessage`` for the message behind
an ``LLVMErrorRef``, ``LLVMDisposeModule``, ``LLVMDisposeBuilder``,
``LLVMContextDispose``, and so on. Ownership is also transferred in
places: a context owns the modules created in it, and an execution
engine takes over the module it was built from, so disposing both
the engine and the module frees the module twice.

**Threads.** The modules that wrap LLVM's long-running work release
the GIL for the duration of the C call, so a pass pipeline, a link
or a JIT compilation running in one thread no longer blocks the
others:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Releases the GIL for
   * - ``analysis``
     - module and function verification
   * - ``bitreader``, ``bitwriter``, ``irreader``
     - parsing and writing bitcode and textual IR
   * - ``linker``
     - ``LLVMLinkModules2``
   * - ``transforms.passbuilder``
     - ``LLVMRunPasses`` and ``LLVMRunPassesOnFunction``
   * - ``targetmachine``
     - target machine creation and emission to file or buffer
   * - ``executionengine``, ``lljit``
     - MCJIT and ORC compilation, materialization and lookup
   * - ``lto``
     - the whole LTO API

Everything else --- ``core`` above all, where a call is typically a
single field access --- keeps the GIL held, because releasing and
reacquiring it costs more than the call it would guard.

.. note::

   Functions that take a callback keep the GIL in every module, so
   a handler can call back into Python. Note that only raw C
   function pointers are accepted here; a :py:mod:`ctypes` callback
   reacquires the GIL itself.

   The bindings do not make LLVM thread-safe. LLVM contexts are not
   shared: two threads must not touch the same ``LLVMContextRef``,
   module or builder concurrently, and the burden of arranging that
   is now yours rather than the GIL's.

.. _sec_llvm_targets:

Listing the installed targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to register LLVM's targets and walk the resulting list.
   * How to look a target up by triple and read its data layout.

Everything target related starts with registration. The three
``LLVMInitializeAll*`` calls populate the target registry, after
which the targets form a linked list that
``LLVMGetFirstTarget`` and ``LLVMGetNextTarget`` walk:

.. literalinclude:: ../../examples/2_Advanced/list_targets.py
   :language: python
   :start-after: [literalinclude-iterate-targets-begin]
   :end-before: [literalinclude-iterate-targets-end]
   :dedent:
   :name: llvm_iterate_targets
   :caption: Register the targets and walk them

Note the disposal in that loop. ``LLVMCopyStringRepOfTargetData``
allocates the data-layout string, as its name says, so the string
goes back through ``LLVMDisposeMessage``; the target and machine
handles belong to LLVM and are not disposed here.

A specific target is found by triple. ``amdgcn-amd-amdhsa`` is the
one the AMD GPU backend registers, and its presence is a good check
that the loaded LLVM has the AMDGPU backend compiled in:

.. literalinclude:: ../../examples/2_Advanced/list_targets.py
   :language: python
   :start-after: [literalinclude-target-from-triple-begin]
   :end-before: [literalinclude-target-from-triple-end]
   :dedent:
   :name: llvm_target_from_triple
   :caption: Look up the AMDGPU target by triple

The full program is ``examples/2_Advanced/list_targets.py``.

.. _sec_llvm_bitcode:

Reading a bitcode file
^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to get a bitcode file into a memory buffer and parse it.
   * How to enumerate the functions of the resulting module.

LLVM reads from memory buffers, so parsing a file is two steps: fill
a buffer from the file, then parse the buffer.

.. literalinclude:: ../../examples/2_Advanced/parse_llvm_bitcode.py
   :language: python
   :start-after: [literalinclude-parse-bitcode-begin]
   :end-before: [literalinclude-parse-bitcode-end]
   :dedent:
   :name: llvm_parse_bitcode
   :caption: Read a bitcode file into a module

The module's functions are another linked list. ``LLVMGetValueName2``
returns the name together with its length, because LLVM's names may
contain embedded null bytes:

.. literalinclude:: ../../examples/2_Advanced/parse_llvm_bitcode.py
   :language: python
   :start-after: [literalinclude-list-functions-begin]
   :end-before: [literalinclude-list-functions-end]
   :dedent:
   :name: llvm_list_functions
   :caption: Enumerate the functions in the module

Run ``examples/2_Advanced/parse_llvm_bitcode.py`` on any bitcode
file, for instance the device library that ships with ROCm at
``<rocm>/amdgcn/bitcode/opencl.bc``. Textual IR goes through
``LLVMParseIRInContext`` from ``rocm.bindings.llvm.c.irreader``
instead, and ``rocm.bindings.llvm.c.bitwriter`` writes both forms
back out.

.. _sec_llvm_build_run:

Building a module and running it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to build a function with the IR builder.
   * How to verify a module without letting LLVM abort the process.
   * How to execute the result through an execution engine.

The IR builder API constructs a function instruction by instruction.
This one adds its two arguments and returns the sum:

.. literalinclude:: ../../examples/2_Advanced/execution_engine_sum.py
   :language: python
   :start-after: [literalinclude-build-module-begin]
   :end-before: [literalinclude-build-module-end]
   :dedent:
   :name: llvm_build_sum
   :caption: Build ``int sum(int, int)`` with the IR builder

Verification is worth doing before anything consumes the module,
and the action argument decides how a defect is reported.
``LLVMAbortProcessAction`` prints and calls ``abort()``, which no
Python ``except`` can catch; ``LLVMReturnStatusAction`` reports the
same defect as a return value:

.. literalinclude:: ../../examples/2_Advanced/execution_engine_sum.py
   :language: python
   :start-after: [literalinclude-verify-module-begin]
   :end-before: [literalinclude-verify-module-end]
   :dedent:
   :name: llvm_verify_module
   :caption: Verify the module before using it

The module can then be executed in-process. The example asks for the
interpreter by name rather than taking whatever
``LLVMCreateExecutionEngineForModule`` picks, because that function
returns an MCJIT engine as soon as some other code in the process
has registered a code generator:

.. literalinclude:: ../../examples/2_Advanced/execution_engine_sum.py
   :language: python
   :start-after: [literalinclude-run-interpreter-begin]
   :end-before: [literalinclude-run-interpreter-end]
   :dedent:
   :name: llvm_run_interpreter
   :caption: Run the function through LLVM's interpreter

Arguments and results cross the boundary as generic values, which is
why the integers go through ``LLVMCreateGenericValueOfInt`` and come
back through ``LLVMGenericValueToInt``. The engine now owns the
module, so the shutdown sequence disposes the engine and the
builder, not the module:

.. literalinclude:: ../../examples/2_Advanced/execution_engine_sum.py
   :language: python
   :start-after: [literalinclude-dispose-begin]
   :end-before: [literalinclude-dispose-end]
   :dedent:
   :name: llvm_dispose_engine
   :caption: Shut down in the right order

The full program is
``examples/2_Advanced/execution_engine_sum.py``; it also writes the
module out as bitcode with ``LLVMWriteBitcodeToFile``.

.. _sec_llvm_passes:

Running an optimization pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: What will I learn?

   * How to run LLVM's new pass manager over a module.
   * How to spell a pipeline, and how to set the options that
     govern it.
   * How to report a pass-manager error without leaking it.

``LLVMRunPasses`` takes a module, a pipeline written in the same
syntax ``opt``'s ``-passes`` flag accepts, an optional target
machine, and an options object. The example
``examples/2_Advanced/llvm_optimize_module.py`` builds a function
that stores an intermediate result to a stack slot, so that the
pipeline has something visible to do:

.. literalinclude:: ../../examples/2_Advanced/llvm_optimize_module.py
   :language: python
   :start-after: [literalinclude-build-module-begin]
   :end-before: [literalinclude-build-module-end]
   :dedent:
   :name: llvm_passes_build_module
   :caption: A module with a stack slot to promote

Running the pipeline is three calls --- create the options, run,
dispose --- with the error handling in between:

.. literalinclude:: ../../examples/2_Advanced/llvm_optimize_module.py
   :language: python
   :start-after: [literalinclude-run-passes-begin]
   :end-before: [literalinclude-run-passes-end]
   :dedent:
   :name: llvm_run_passes
   :caption: Run ``default<O2>`` over the module

Passing ``None`` as the target machine gives LLVM a null
``LLVMTargetMachineRef``, which selects a target-independent
pipeline. Supply a real one --- built with
``LLVMCreateTargetMachine`` as in :ref:`sec_llvm_targets` --- when
the passes should see a target's data layout and cost model, which
is what ``numba.hip`` does for ``amdgcn-amd-amdhsa``.

The options object carries the pipeline's knobs, one
``LLVMPassBuilderOptionsSet*`` function per knob. ``VerifyEach`` is
tempting but dangerous: like the ``verify`` pass, it reports a
broken module by aborting the process. Verify by return value first,
as shown in the previous section, and the pipeline never has the
opportunity.

Printing the module before and after shows the effect. ``str(...)``
on the returned :py:obj:`~.types.CStr` yields the
IR text, and the buffer belongs to the caller:

.. literalinclude:: ../../examples/2_Advanced/llvm_optimize_module.py
   :language: python
   :start-after: [literalinclude-print-module-begin]
   :end-before: [literalinclude-print-module-end]
   :dedent:
   :name: llvm_print_module
   :caption: Print a module's IR and dispose the buffer

With ``default<O2>``, the stack slot is promoted to a register and
the function collapses to the two arithmetic instructions:

.. code-block:: text

   --- before ---
   define i32 @square_of_sum(i32 %0, i32 %1) {
   entry:
     %slot = alloca i32, align 4
     %sum = add i32 %0, %1
     store i32 %sum, ptr %slot, align 4
     %reloaded = load i32, ptr %slot, align 4
     %square = mul i32 %reloaded, %reloaded
     ret i32 %square
   }
   --- after 'default<O2>' ---
   define i32 @square_of_sum(i32 %0, i32 %1) local_unnamed_addr #0 {
   entry:
     %sum = add i32 %1, %0
     %square = mul i32 %sum, %sum
     ret i32 %square
   }

``numba.hip`` wraps the same three calls in
``numba.hip.amdgcn.AMDGPUTargetMachine.optimize_module``, which
accepts IR text, bitcode or a module handle and adds a target
machine for the GPU it compiles for.

.. _sec_llvm_handoff:

Handing IR back to HIPRTC and AMD COMGR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bitcode produced through these bindings is ordinary bitcode, so it
crosses back into the first half of this chapter unchanged:

* HIPRTC's linker takes it as
  :py:obj:`~.hiprtcJITInputType.HIPRTC_JIT_INPUT_LLVM_BITCODE`
  input, next to bitcode from HIP sources compiled with
  ``-fgpu-rdc``; see :ref:`sec_jit_hiprtc_llvm_ir`.
* :py:obj:`rocm.comgr` consumes bitcode at
  :py:obj:`~rocm.comgr.compile_bc_to_hsa` and produces it at
  :py:obj:`~rocm.comgr.compile_hip_to_bc`; see
  :ref:`sec_jit_comgr_hip_to_bc`.

Whatever the direction, the target matters: bitcode meant for an AMD
GPU has to have been produced for the right AMDGPU triple and
architecture. The LLVM bindings will happily parse and transform
bitcode for a target the GPU cannot run.

.. _sec_jit_further_examples:

Further examples
----------------

The ``examples/2_Advanced/`` directory contains additional recipes
that build on the techniques above:

* ``hiprtc_amd_comgr_hip_to_hsa.py`` and
  ``hiprtc_amd_comgr_hsa_to_code_obj.py`` --- run the
  HIP --> HSA --> code-object pipeline by hand, mixing HIPRTC and
  COMGR stages.
* ``hiprtc_jit_with_llvm_ir.py`` --- a minimal end-to-end
  variant of :ref:`sec_jit_hiprtc_llvm_ir` (``gfx90a`` only).
* ``amd_comgr_disassemble_amdgpu_code_obj.py`` /
  ``amd_comgr_disassemble_amdgpu_program.py`` --- disassemble
  an existing code object or a HIP program from disk, without
  any HIPRTC involvement.
* ``amd_comgr_parse_amd_hsa_kernel_descriptor.py`` --- decode
  the 64-byte AMD HSA kernel descriptor block embedded in a
  code object.
* ``amd_comgr_hip_to_llvm_ir.py`` and ``hiprtc_hip_to_llvm_ir.py``
  --- print the bitcode that COMGR and HIPRTC produce as readable
  IR, by parsing it with ``LLVMParseBitcode2`` and printing it with
  ``LLVMPrintModuleToString``.
* ``list_targets.py``, ``parse_llvm_bitcode.py``,
  ``execution_engine_sum.py`` and ``llvm_optimize_module.py`` ---
  the LLVM-C programs the second half of this chapter is built
  from, each runnable on its own.
