rocm.bindings.util.paths
========================

.. py:module:: rocm.bindings.util.paths

.. autoapi-nested-parse::

   ROCm library path resolution utilities.

   This module provides cross-platform library path discovery for ROCm libraries,
   supporting both traditional installations (/opt/rocm) and TheRock/rocm_sdk
   package-based installations.

   The get_library_path() function is designed for lazy evaluation - it should be
   called when a library is first accessed, not at module import time, to preserve
   the lazy loading behavior of the ROCm bindings.



Functions
---------

.. autoapisummary::

   rocm.bindings.util.paths.get_clang_resource_dir
   rocm.bindings.util.paths.get_library_path


Module Contents
---------------

.. py:function:: get_clang_resource_dir(libclang_file: Optional[str] = None) -> Optional[str]

   Directory holding clang's own headers, as '-resource-dir' expects it.

   Anything that parses HIP or C++ sources through libclang needs this
   directory, and where it sits depends on the installation, so the same
   layout knowledge get_library_path carries applies here.

   The resource directory belongs to a specific libclang and normally sits
   beside it, at <dir>/clang/<version>. That holds wherever shared libraries
   and their data share a directory, which is every platform but Windows:
   there the DLLs live in a bin directory while the resource directory stays
   in the sibling lib, so both are searched.

   Args:
       libclang_file: The libclang to resolve the directory for. Defaults to
                      the one get_library_path finds.

   Returns:
       The resource directory, or None if no installation could be found.


.. py:function:: get_library_path(shortname: str, bundled_location: Optional[pathlib.Path] = None) -> bytes

   Get platform-appropriate library path for ROCm library (lazy evaluation).

   This function uses a fallback chain to locate ROCm libraries across different
   installation methods and platforms:

   1. Bundled location - Recursively search rocm package for library
   2. rocm_sdk.find_libraries() - TheRock Python package installation.
      For the LLVM toolchain (clang/LLVM), which rocm_sdk does not register,
      anchor on 'amdhip64' and resolve the sibling <core>/lib/llvm/lib, then
      fall back to rocm_sdk._devel.get_devel_root() (the rocm-sdk-devel tree).
   3. ROCM_PATH / ROCM_HOME environment variable - Traditional install.
      On Unix, LLVM and clang resolve under <rocm>/llvm/lib and a versioned
      soname is allowed. On Windows the DLLs live in <rocm>/bin (LLVM in
      <rocm>/lib/llvm/bin) under names that are not simply <shortname>.dll,
      so the tree is scanned for the candidates _windows_dll_candidates lists.
   4. Basename fallback - Rely on system loader (LD_LIBRARY_PATH on Linux,
      DYLD_LIBRARY_PATH on macOS). On Windows PATH is walked here instead,
      because the bare name the loader would be given is often not the name
      on disk.

   This is designed to be called lazily when a library is first accessed,
   preserving the lazy loading behavior of the bindings.

   Args:
       shortname: Library short name without platform-specific prefix/suffix
                  Examples: 'amdhip64', 'hiprtc', 'hipblas', 'hipsolver', 'LLVM'
       bundled_location: Optional path to bundled library directory.
                        If None, recursively searches from rocm package root.

   Returns:
       Absolute path to library as bytes, or basename if not found.
       The returned bytes can be passed directly to open_library().

   Platform-specific behavior:
       Windows:
           - DLL names differ from the other platforms: the HIP runtime and
             hipRTC are version-suffixed (amdhip64_7.dll, hiprtc0715.dll) and a
             few libraries keep the Unix 'lib' prefix (libhipblaslt.dll). See
             _windows_dll_candidates.
           - Traditional HIP SDK: DLLs are in C:\Windows\System32 (installed
             by the GPU driver), found by walking PATH
           - Tarball / TheRock unpacked: DLLs are in <ROCM_PATH>\bin
           - rocm_sdk (TheRock): DLLs are in site-packages\_rocm_sdk_core\bin
             The rocm_sdk.find_libraries() API returns the full path, already
             carrying the correct Windows name

       macOS (Darwin):
           - Traditional: Libraries in /opt/rocm/lib or ROCM_PATH/lib (.dylib extension)
           - rocm_sdk (TheRock): Libraries in site-packages/_rocm_sdk_core/lib
             The rocm_sdk.find_libraries() API returns the full path

       Linux:
           - Traditional: Libraries in /opt/rocm/lib or ROCM_PATH/lib (.so extension)
           - rocm_sdk (TheRock): Libraries in site-packages/_rocm_sdk_core/lib
             The rocm_sdk.find_libraries() API returns the full path

   Examples:
       >>> # Get path for HIP runtime
       >>> path = get_library_path('amdhip64')
       >>> # path might be:
       >>> # - b'/path/to/site-packages/_rocm_sdk_core/lib/libamdhip64.so' (rocm_sdk on Linux)
       >>> # - b'/path/to/site-packages/_rocm_sdk_core/lib/libamdhip64.dylib' (rocm_sdk on macOS)
       >>> # - b'C:\\Users\\...\\site-packages\\_rocm_sdk_core\\bin\\amdhip64.dll' (rocm_sdk on Windows)
       >>> # - b'/opt/rocm/lib/libamdhip64.so' (traditional Linux)
       >>> # - b'/opt/rocm/lib/libamdhip64.dylib' (traditional macOS)
       >>> # - b'amdhip64.dll' (traditional Windows, relies on System32)


