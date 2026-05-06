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

# Find LLVM using ROCm's LLVM installation
message(STATUS "=== bundled/libllvm/CMakeLists.txt: Starting libLLVM detection ===")
if(DEFINED ROCM_PATH)
  set(LLVM_DIR "${ROCM_PATH}/llvm/lib/cmake/llvm}")
  message(STATUS "ROCM_PATH is set to: ${ROCM_PATH}")
endif()

find_package(LLVM REQUIRED CONFIG)

# Options to control bundling behavior (can be set via parent CMakeLists.txt)
option(HIP_PYTHON_BUNDLE_LIBLLVM "Bundle libLLVM.so in rocm-bindings-compiler wheel" ON)
option(HIP_PYTHON_FORCE_BUILD_LIBLLVM
  "Always build libLLVM.so from LLVM static archives (--whole-archive) instead of\
 reusing the system shared library. Yields a self-contained library that exports\
 every LLVM symbol (e.g. LLVMInitializeAllTargetInfos), at the cost of a much\
 longer link step and a larger wheel. Implies HIP_PYTHON_BUNDLE_LIBLLVM=ON."
  OFF)

# FORCE_BUILD requires BUNDLE: building libLLVM.so without shipping it would
# leave the extension modules without anything to dlopen at runtime.
if(HIP_PYTHON_FORCE_BUILD_LIBLLVM AND NOT HIP_PYTHON_BUNDLE_LIBLLVM)
  message(FATAL_ERROR
    "HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON requires HIP_PYTHON_BUNDLE_LIBLLVM=ON, "
    "but HIP_PYTHON_BUNDLE_LIBLLVM was explicitly set to OFF. "
    "Either enable bundling, or disable forced build.")
endif()

# Check if libLLVM.so exists in ROCm LLVM library directory.
# These flags are propagated to PARENT_SCOPE at the end of this file so the
# parent rocm-bindings-compiler/CMakeLists.txt can wire dependencies on the
# bundled rocmllvm target.
set(BUNDLE_SYSTEM_LIBLLVM OFF)
set(BUILD_BUNDLED_LIBLLVM OFF)

if(HIP_PYTHON_BUNDLE_LIBLLVM)
  if(HIP_PYTHON_FORCE_BUILD_LIBLLVM)
    message(STATUS "HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON - will build and bundle libLLVM.so from static archives")
    set(BUILD_BUNDLED_LIBLLVM ON)
  elseif(DEFINED ROCM_PATH)
    if(EXISTS "${ROCM_PATH}/llvm/lib")
      # Look for libLLVM.so in ROCm LLVM library directory
      find_library(SYSTEM_LIBLLVM
        NAMES LLVM
        PATHS "${ROCM_PATH}/llvm/lib" ${LLVM_LIBRARY_DIRS}
        NO_DEFAULT_PATH
      )
      if(SYSTEM_LIBLLVM)
        message(STATUS "Found system libLLVM.so at ${SYSTEM_LIBLLVM} - will copy to wheel")
        set(BUNDLE_SYSTEM_LIBLLVM ON)
      else()
        message(STATUS "libLLVM.so not found in ${ROCM_PATH}/llvm/lib - will build and bundle")
        set(BUILD_BUNDLED_LIBLLVM ON)
      endif()
    else()
      message(STATUS "${ROCM_PATH}/llvm/lib doesn't exist - will build and bundle libLLVM.so")
      set(BUILD_BUNDLED_LIBLLVM ON)
    endif()
  else()
    message(STATUS "ROCM_PATH not set - will build and bundle libLLVM.so")
    set(BUILD_BUNDLED_LIBLLVM ON)
  endif()
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM library dirs: ${LLVM_LIBRARY_DIRS}")

# Install system libLLVM.so by copying it
if(BUNDLE_SYSTEM_LIBLLVM)
  # SYSTEM_LIBLLVM may be a symlink; resolve to the real file so we copy
  # the actual shared library (not a dangling symlink) into the wheel.
  file(REAL_PATH "${SYSTEM_LIBLLVM}" SYSTEM_LIBLLVM_REAL)
  install(FILES ${SYSTEM_LIBLLVM_REAL}
    DESTINATION rocm/bindings/llvm
    RENAME libLLVM.so
    COMPONENT rocm-bindings-compiler
  )
  message(STATUS "Will install system libLLVM.so from ${SYSTEM_LIBLLVM_REAL}")
endif()

# Only build bundled library if system libLLVM.so not found
if(BUILD_BUNDLED_LIBLLVM)

# Get llvm-config path for querying library files
find_program(LLVM_CONFIG_EXECUTABLE
  NAMES llvm-config
  PATHS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_INSTALL_PREFIX}/bin
  NO_DEFAULT_PATH
)

if(NOT LLVM_CONFIG_EXECUTABLE)
  message(FATAL_ERROR "llvm-config not found")
endif()

message(STATUS "Using llvm-config: ${LLVM_CONFIG_EXECUTABLE}")

# Query llvm-config for static library files. --link-static is required:
# without it, an LLVM build configured with LLVM_BUILD_LLVM_DYLIB=ON returns
# only the shared libLLVM, which would defeat the --whole-archive link below
# (re-linking the same symbols, not adding the missing inline-helper ones).
execute_process(
  COMMAND ${LLVM_CONFIG_EXECUTABLE} --link-static --libfiles
  OUTPUT_VARIABLE LLVM_LIBFILES
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE LLVM_LIBFILES_RESULT
)
if(NOT LLVM_LIBFILES_RESULT EQUAL 0)
  message(FATAL_ERROR
    "llvm-config --link-static --libfiles failed (exit=${LLVM_LIBFILES_RESULT}). "
    "The LLVM installation at ${LLVM_INSTALL_PREFIX} may not provide static archives, "
    "which are required to build a self-contained libLLVM.so.")
endif()

# Convert space-separated string to list
string(REPLACE " " ";" LLVM_LIBFILES_LIST "${LLVM_LIBFILES}")

message(STATUS "LLVM library files: ${LLVM_LIBFILES_LIST}")

# Query for additional flags
execute_process(
  COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
  OUTPUT_VARIABLE LLVM_CXXFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags
  OUTPUT_VARIABLE LLVM_LDFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Use --link-static so we get the full transitive system-libs needed by the
# static archives (zlib, zstd, ncurses, libxml2, ...) — without it, llvm-config
# may report only the deps of the shared libLLVM, leaving the bundled libLLVM.so
# with undefined symbols (e.g. crc32 from zlib).
execute_process(
  COMMAND ${LLVM_CONFIG_EXECUTABLE} --link-static --system-libs
  OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# TheRock-style ROCm packages link against shimmed system deps
# (librocm_sysdeps_z.so.1, librocm_sysdeps_zstd.so.1.5.7, ...) that live in a
# non-standard libdir. Add it to the link search path so the bundled libLLVM.so
# can resolve those references.
set(ROCM_SYSDEPS_LIBDIR "${ROCM_PATH}/lib/rocm_sysdeps/lib")
if(EXISTS "${ROCM_SYSDEPS_LIBDIR}")
  message(STATUS "Adding ROCm sysdeps libdir to link path: ${ROCM_SYSDEPS_LIBDIR}")
endif()

# llvm-config emits the shimmed deps as `-lrocm_sysdeps_z.so.1`, but `-l<name>`
# normally resolves to `lib<name>.so`. Rewrite those entries to ld's literal
# filename syntax (`-l:librocm_sysdeps_z.so.1`) so the linker finds the actual
# versioned shared library.
string(REGEX REPLACE "-l(rocm_sysdeps_[^ ]+\\.so[^ ]*)" "-l:lib\\1"
       LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS}")

# Create shared library - name it libLLVM.so for easier discovery
add_library(rocmllvm SHARED Target.cpp)
set_target_properties(rocmllvm PROPERTIES OUTPUT_NAME "LLVM")

target_include_directories(rocmllvm
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

# Apply LLVM C++ flags
separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS}")
target_compile_options(rocmllvm PRIVATE ${LLVM_CXXFLAGS_LIST})

# Link with LLVM libraries using --whole-archive (mimics Makefile)
# This ensures all symbols from static libs are included in the shared lib.
# CMake separates target_link_options from target_link_libraries on the link
# line — sandwiching the library list with --whole-archive / --no-whole-archive
# via target_link_options would put the toggles adjacent (no-op). Pass them
# inline with the libraries via target_link_libraries instead.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # -Bsymbolic binds global symbols within this library to prevent symbol
  # interposition from other libraries.
  target_link_options(rocmllvm PRIVATE -Wl,-Bsymbolic)
  target_link_libraries(rocmllvm PRIVATE
    -Wl,--whole-archive
    ${LLVM_LIBFILES_LIST}
    -Wl,--no-whole-archive
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # macOS: No -Bsymbolic equivalent needed
  # Two-level namespaces (default) provide symbol isolation
  # Use -all_load equivalent for whole-archive behavior
  target_link_options(rocmllvm PRIVATE -Wl,-all_load)
  target_link_libraries(rocmllvm PRIVATE ${LLVM_LIBFILES_LIST})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # Windows: No -Bsymbolic equivalent needed
  # DLLs use per-DLL symbol namespaces by design
  # /WHOLEARCHIVE for including all symbols from static libs
  foreach(lib ${LLVM_LIBFILES_LIST})
    target_link_options(rocmllvm PRIVATE /WHOLEARCHIVE:${lib})
  endforeach()
  target_link_libraries(rocmllvm PRIVATE ${LLVM_LIBFILES_LIST})
endif()

# Apply LLVM linker flags and system libs
separate_arguments(LLVM_LDFLAGS_LIST UNIX_COMMAND "${LLVM_LDFLAGS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST UNIX_COMMAND "${LLVM_SYSTEM_LIBS}")
target_link_options(rocmllvm PRIVATE ${LLVM_LDFLAGS_LIST})
if(EXISTS "${ROCM_SYSDEPS_LIBDIR}")
  target_link_options(rocmllvm PRIVATE "-L${ROCM_SYSDEPS_LIBDIR}")
endif()
target_link_libraries(rocmllvm PRIVATE ${LLVM_SYSTEM_LIBS_LIST})

# Set RPATH for finding LLVM libraries.
# $ORIGIN -> bundled libs in same dir; LLVM_LIBRARY_DIRS -> system LLVM libdir;
# ROCM_SYSDEPS_LIBDIR -> TheRock-style shimmed system deps.
set_target_properties(rocmllvm PROPERTIES
  INSTALL_RPATH "$ORIGIN:${LLVM_LIBRARY_DIRS}:${ROCM_SYSDEPS_LIBDIR}"
  BUILD_RPATH "$ORIGIN:${LLVM_LIBRARY_DIRS}:${ROCM_SYSDEPS_LIBDIR}"
)

# Install to package directory (bundled with wheel)
# IMPORTANT: Set default component to ensure this gets installed with the package
# Library will be named libLLVM.so (via OUTPUT_NAME property)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rocm-bindings-compiler)
install(TARGETS rocmllvm
  LIBRARY DESTINATION rocm/bindings/llvm
  RUNTIME DESTINATION rocm/bindings/llvm  # For Windows DLL
  COMPONENT rocm-bindings-compiler
)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME)

# Install README explaining the bundled libLLVM.so
if(BUNDLE_SYSTEM_LIBLLVM)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LIBLLVM_README.txt"
"This directory contains libLLVM.so copied from the system ROCm LLVM installation.
Source: ${SYSTEM_LIBLLVM}

This ensures the wheel is self-contained and does not rely on system RPATH.
")
else()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LIBLLVM_README.txt"
"This directory contains a bundled libLLVM.so built from ROCm's LLVM.
It is included because the system libLLVM.so was not found in the
standard ROCm LLVM library directory (${LLVM_LIBRARY_DIRS}).

The bundled library is built with --whole-archive to include all LLVM
components needed by the Python bindings.
")
endif()

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/LIBLLVM_README.txt"
  DESTINATION rocm/bindings/llvm
  COMPONENT rocm-bindings-compiler
)
endif() # BUILD_BUNDLED_LIBLLVM

# Propagate detection results to the parent CMakeLists.txt so it can wire
# the cython modules and package target against the bundled rocmllvm.
set(BUNDLE_SYSTEM_LIBLLVM ${BUNDLE_SYSTEM_LIBLLVM} PARENT_SCOPE)
set(BUILD_BUNDLED_LIBLLVM ${BUILD_BUNDLED_LIBLLVM} PARENT_SCOPE)
