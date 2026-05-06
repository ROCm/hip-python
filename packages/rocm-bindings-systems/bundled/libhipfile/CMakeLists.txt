# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Optional bundling of libhipfile.so. Mirrors the libLLVM bundling pattern in
# rocm-bindings-compiler/bundled/libllvm/. Install the resolved library into
# the wheel and point the cython modules at it via $ORIGIN RPATH so they
# don't need a system hipfile installation at runtime.
#
# Inputs (set by the parent rocm-bindings-systems/CMakeLists.txt):
#   HIP_PYTHON_BUNDLE_LIBHIPFILE  (option, default OFF)
#   HIP_PYTHON_HIPFILE_AVAILABLE  (TRUE if find_package(hipfile) succeeded)
#   HIPFILE_LIBRARY_PATH          (path from hip::hipfile target)
#
# Targets `rocm_bindings_systems_cyhipfile` and `rocm_bindings_hipfile`
# MUST already exist when this file is included — i.e. the parent
# CMakeLists.txt MUST add_subdirectory(bundled/libhipfile) AFTER the
# foreach loop that creates those targets.
#
# This file is a no-op when bundling is disabled; the parent always
# add_subdirectory()s it unconditionally so the option remains live.

if(HIP_PYTHON_BUNDLE_LIBHIPFILE)
  if(NOT HIP_PYTHON_HIPFILE_AVAILABLE)
    message(WARNING "HIP_PYTHON_BUNDLE_LIBHIPFILE=ON but hipfile package was not found — nothing to bundle")
  elseif(NOT HIPFILE_LIBRARY_PATH OR NOT EXISTS "${HIPFILE_LIBRARY_PATH}")
    message(WARNING "HIP_PYTHON_BUNDLE_LIBHIPFILE=ON but hip::hipfile IMPORTED_LOCATION was not resolvable — nothing to bundle")
  else()
    # IMPORTED_LOCATION may be a versioned symlink; resolve it so we copy the
    # actual shared object (not a dangling link) into the wheel.
    file(REAL_PATH "${HIPFILE_LIBRARY_PATH}" HIPFILE_LIBRARY_REAL)
    install(FILES ${HIPFILE_LIBRARY_REAL}
      DESTINATION rocm/bindings
      RENAME libhipfile.so
      COMPONENT rocm-bindings-systems
    )
    message(STATUS "Bundling libhipfile.so from ${HIPFILE_LIBRARY_REAL}")

    # Point the hipfile extension modules at the bundled library. $ORIGIN
    # is the directory containing the .so at runtime — i.e. rocm/bindings/.
    if(TARGET rocm_bindings_systems_cyhipfile)
      set_target_properties(rocm_bindings_systems_cyhipfile PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
      )
    endif()
    if(TARGET rocm_bindings_hipfile)
      set_target_properties(rocm_bindings_hipfile PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
      )
    endif()
  endif()
endif()
