rocm.comgr.amdhsa_kernel_directives
===================================

.. py:module:: rocm.comgr.amdhsa_kernel_directives

.. autoapi-nested-parse::

   AMD HSA kernel directives.

   The data tables below (``arch_supported_expressions``,
   ``arch_unsupported_expressions``, ``directives``) are scraped from
   the upstream LLVM AMD GPU usage docs. The JavaScript snippet that
   performs the scrape is kept as a plain ``#``-comment block right
   below this docstring (out of the docstring on purpose: a markdown
   fenced JS block and the JS template-literal backticks confuse the
   RST/MyST parser used by sphinx-autoapi). To regenerate the data:
   open <https://llvm.org/docs/AMDGPUUsage.html>, paste the snippet
   into the browser console, and replace the data tables with the
   captured output.



Attributes
----------

.. autoapisummary::

   rocm.comgr.amdhsa_kernel_directives.p_amdgpu_arch
   rocm.comgr.amdhsa_kernel_directives.arch_supported_expressions
   rocm.comgr.amdhsa_kernel_directives.arch_unsupported_expressions
   rocm.comgr.amdhsa_kernel_directives.directives


Functions
---------

.. autoapisummary::

   rocm.comgr.amdhsa_kernel_directives.split_amdgpu_arch
   rocm.comgr.amdhsa_kernel_directives.is_amdgpu_arch_supported
   rocm.comgr.amdhsa_kernel_directives.is_amdgpu_arch_unsupported
   rocm.comgr.amdhsa_kernel_directives.iter_supported_directives


Module Contents
---------------

.. py:data:: p_amdgpu_arch

.. py:function:: split_amdgpu_arch(amdgpu_arch)

.. py:function:: is_amdgpu_arch_supported(supported_expression: str, amdgpu_arch: str) -> str

.. py:function:: is_amdgpu_arch_unsupported(unsupported_expression: str, amdgpu_arch: str) -> str

.. py:function:: iter_supported_directives(amdgpu_arch: str, features: list[str]) -> Generator[dict[str, object]]

.. py:data:: arch_supported_expressions
   :value: ['GFX6-GFX12', 'GFX6-GFX10', 'GFX12.5', 'GFX10-GFX12', 'GFX1250+', 'GFX942', 'GFX11-GFX12',...


.. py:data:: arch_unsupported_expressions
   :value: ['GFX942']


.. py:data:: directives

