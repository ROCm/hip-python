rocm.comgr.amd_hsa_kernel_descriptor
====================================

.. py:module:: rocm.comgr.amd_hsa_kernel_descriptor


Attributes
----------

.. autoapisummary::

   rocm.comgr.amd_hsa_kernel_descriptor.p_amdgpu_arch


Classes
-------

.. autoapisummary::

   rocm.comgr.amd_hsa_kernel_descriptor.AMDHSAKernelDescriptor


Functions
---------

.. autoapisummary::

   rocm.comgr.amd_hsa_kernel_descriptor.parse_amdgpu_code_obj_kernel_descriptor


Module Contents
---------------

.. py:data:: p_amdgpu_arch

.. py:class:: AMDHSAKernelDescriptor

   A class for accessing / rendering kernel-related information stored in an
   AMD HSA code object v6.

   Derived from struct `kernel_descript_t` in file:

   <https://github.com/ROCm/llvm-project/blob/a53433cf1b9f533b51b73ea82d69f78041e40f93/llvm/include/llvm/Support/AMDHSAKernelDescriptor.h>

   Note:
       We want to highlight the following comment above the struct:
       `// Kernel descriptor. Must be kept backwards compatible.`,
       which implies that the main layout (`group_types`)
       will likely not change.

   Note:
       We preprocessed ``AMDHSAKernelDescriptor.h`` via

       ```shell
       g++ -E AMDHSAKernelDescriptor.h -o AMDHSAKernelDescriptor.h.i
       ```

       and then collected the enum values and transformed the keys to lower
       case. This yielded the class member `_group_entry_coordinates`.
       The iterator '_iterate_group_entries' has a lot of logic to support
       maintenance via this approach.


   .. py:method:: get_possible_field_names()


   .. py:method:: create_type(amdgpu_arch: str, features: Iterable[str] = []) -> ctypes.Structure
      :staticmethod:



.. py:function:: parse_amdgpu_code_obj_kernel_descriptor(code_symbol: bytes | bytearray, amdgpu_arch: str)

   Parse kernel descriptor symbol extracted from v6 code object.

   Args:
       code_symbol (bytes|bytearray):
           The bytes of the kernel descriptor code object.
       amdgpu_arch (`str`):
           AMD GPU architecture. An expression like 'gfx90a' or 'gfx1201'.
           Assumes that any feature flags such as `:xnack+` have been stripped
           off.

   Returns:
       A ctypes.Structure with fields for the given architecture.


