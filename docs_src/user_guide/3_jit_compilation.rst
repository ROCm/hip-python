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

JIT Compilation with HIPRTC and AMD COMGR
=========================================

This chapter shows how to compile, link, and inspect HIP kernels at
runtime. The basic ``hiprtc`` flow --- "compile a HIP source string,
load the resulting code object as a module, launch a kernel" --- is
covered in :ref:`sec_launching_kernels`. Here we go beyond that:

- linking multiple translation units (separable compilation),
- linking HIP code with externally supplied LLVM IR,
- using **AMD COMGR** as a lower-level alternative to HIPRTC for
  pipeline stages that HIPRTC does not expose (e.g. emitting HSA
  assembly, parsing code-object metadata, disassembling kernels).

Two Python-level entry points are used throughout:

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

.. note::

   The example files referenced below define a ``hip_check`` helper
   that raises :py:obj:`RuntimeError` on a non-success return
   status (and an analogous ``llvm_check``). The page snippets
   below omit those helpers --- if you copy a snippet, wrap each
   ``hip.``/``hiprtc.`` call in your own status check.

.. _sec_jit_hiprtc_linking:

Linking HIP translation units with HIPRTC
-----------------------------------------

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
--------------------------------------------

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
LLVM IR module that defines a missing ``__device__`` function.

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
-----------------------------------------

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
yourself:

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
-------------------------------------------

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
-------------------------------------

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

.. _sec_jit_further_examples:

Further examples
----------------

The ``examples/2_Advanced/`` directory contains additional JIT
recipes that build on the techniques above:

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
* ``execution_engine_sum.py`` --- bind LLVM's MCJIT execution
  engine via the LLVM-C bindings shipped in
  ``rocm.bindings.llvm.c``.
