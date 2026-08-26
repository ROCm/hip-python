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

# Reuse the LLVM the parent already located. LLVM_DIR is cached by that
# find_package, so this resolves to the same installation instead of
# hardcoding a path of its own: a fixed guess and the parent's choice can
# disagree, and on Windows neither is reliably present.
message(STATUS "=== bundled/libllvm/CMakeLists.txt: Starting libLLVM detection ===")
find_package(LLVM REQUIRED CONFIG)

option(HIP_PYTHON_BUNDLE_LIBLLVM "Bundle libLLVM.so in rocm-bindings-compiler wheel" ${HIP_PYTHON_BUNDLE_LIBLLVM_DEFAULT})
option(HIP_PYTHON_FORCE_BUILD_LIBLLVM
  "Always build the shared LLVM from LLVM's static archives instead of reusing\
 the system shared library. Yields a self-contained library that also carries\
 the header-inline helpers (e.g. LLVMInitializeAllTargetInfos), at the cost of a\
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

# The filename the bindings will look for at runtime. rocm.bindings.util.paths
# resolves the short name 'LLVM' with the platform's prefix and suffix, so the
# bundled copy has to land under that exact name.
if(WIN32)
  set(HIP_PYTHON_LIBLLVM_NAME "LLVM.dll")
elseif(APPLE)
  set(HIP_PYTHON_LIBLLVM_NAME "libLLVM.dylib")
else()
  set(HIP_PYTHON_LIBLLVM_NAME "libLLVM.so")
endif()

if(HIP_PYTHON_BUNDLE_LIBLLVM)
  if(HIP_PYTHON_FORCE_BUILD_LIBLLVM)
    message(STATUS "HIP_PYTHON_FORCE_BUILD_LIBLLVM=ON - will build and bundle ${HIP_PYTHON_LIBLLVM_NAME} from static archives")
    set(BUILD_BUNDLED_LIBLLVM ON)
  else()
    # Look for a shared LLVM belonging to the installation found above.
    # LLVM_LIBRARY_DIRS comes from LLVMConfig.cmake, so it tracks whichever
    # layout was discovered; ${ROCM_PATH}/llvm/lib is retained for the
    # traditional Linux tree. On Windows the loadable artifact is a DLL and
    # lives in a bin directory — find_library would match the LLVM.lib import
    # library instead, which is not something the bindings can dlopen.
    if(WIN32)
      find_file(SYSTEM_LIBLLVM
        NAMES "${HIP_PYTHON_LIBLLVM_NAME}"
        PATHS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_LIBRARY_DIRS} "${ROCM_PATH}/bin"
        NO_DEFAULT_PATH
      )
    else()
      find_library(SYSTEM_LIBLLVM
        NAMES LLVM
        PATHS ${LLVM_LIBRARY_DIRS} "${ROCM_PATH}/llvm/lib"
        NO_DEFAULT_PATH
      )
    endif()
    if(SYSTEM_LIBLLVM)
      message(STATUS "Found system ${HIP_PYTHON_LIBLLVM_NAME} at ${SYSTEM_LIBLLVM} - will copy to wheel")
      set(BUNDLE_SYSTEM_LIBLLVM ON)
    else()
      message(STATUS "${HIP_PYTHON_LIBLLVM_NAME} not found near ${LLVM_DIR} - will build and bundle")
      set(BUILD_BUNDLED_LIBLLVM ON)
    endif()
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
    RENAME "${HIP_PYTHON_LIBLLVM_NAME}"
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

# Query llvm-config for the static archives. --link-static is required: without
# it, an LLVM build configured with LLVM_BUILD_LLVM_DYLIB=ON returns only the
# shared libLLVM, which would defeat the link below (re-linking the same
# symbols, not adding the missing inline-helper ones).
#
# Windows takes --libnames plus --libdir rather than --libfiles: llvm-config
# separates entries with a single space, and an absolute Windows path may
# contain one ("C:/Program Files/..."), so --libfiles cannot be split back into
# a list. Library file names cannot contain spaces, so joining them onto
# --libdir is unambiguous.
if(WIN32)
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
    OUTPUT_VARIABLE LLVM_STATIC_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLVM_LIBDIR_RESULT
  )
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --link-static --libnames all
    OUTPUT_VARIABLE LLVM_LIBNAMES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLVM_LIBFILES_RESULT
  )
  if(LLVM_LIBDIR_RESULT EQUAL 0 AND LLVM_LIBFILES_RESULT EQUAL 0)
    file(TO_CMAKE_PATH "${LLVM_STATIC_LIBDIR}" LLVM_STATIC_LIBDIR)
    string(REPLACE " " ";" LLVM_LIBNAMES_LIST "${LLVM_LIBNAMES}")
    set(LLVM_LIBFILES_LIST "")
    foreach(_libname ${LLVM_LIBNAMES_LIST})
      list(APPEND LLVM_LIBFILES_LIST "${LLVM_STATIC_LIBDIR}/${_libname}")
    endforeach()
  endif()
else()
  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --link-static --libfiles
    OUTPUT_VARIABLE LLVM_LIBFILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLVM_LIBFILES_RESULT
  )
  string(REPLACE " " ";" LLVM_LIBFILES_LIST "${LLVM_LIBFILES}")
endif()

if(NOT LLVM_LIBFILES_RESULT EQUAL 0)
  message(FATAL_ERROR
    "llvm-config --link-static failed (exit=${LLVM_LIBFILES_RESULT}). "
    "The LLVM installation at ${LLVM_INSTALL_PREFIX} may not provide static archives, "
    "which are required to build a self-contained ${HIP_PYTHON_LIBLLVM_NAME}.")
endif()

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

# On Windows llvm-config reports zlib (as zs.lib) but not the zstd archive that
# LLVMSupport references just as unconditionally, so the link fails on
# ZSTD_compress2 and friends unless it is added here. Both live under the
# sysdeps libdir added to the search path above, which llvm-config does not
# mention either.
if(WIN32 AND NOT LLVM_SYSTEM_LIBS MATCHES "zstd")
  find_library(LLVM_ZSTD_LIBRARY
    NAMES zstd_static zstd
    PATHS "${ROCM_SYSDEPS_LIBDIR}" ${LLVM_LIBRARY_DIRS}
    NO_DEFAULT_PATH
  )
  if(LLVM_ZSTD_LIBRARY)
    message(STATUS "Adding zstd for LLVMSupport: ${LLVM_ZSTD_LIBRARY}")
    string(APPEND LLVM_SYSTEM_LIBS " ${LLVM_ZSTD_LIBRARY}")
  endif()
endif()

# Windows exports are opt-in, so the DLL needs a module-definition file.
#
# ELF gives every global symbol in an archive default visibility, which is why
# the --whole-archive link below produces a fully populated libLLVM.so from the
# very same components. A PE image instead exports only what the objects marked
# with __declspec(dllexport), and ROCm's LLVM is compiled with the export
# annotations off (llvm/Config/llvm-config.h leaves
# LLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS undefined), so its archives carry no
# export directives whatsoever: /WHOLEARCHIVE alone yields a DLL with no export
# table, which loads and offers nothing.
#
# Naming the symbols is the remaining option, and the C API is the subset worth
# naming: it is unmangled, it is what the bindings and numba-hip call, and it is
# small enough to fit a PE export table (~1,200 entries, against 65,535
# ordinals; the full symbol set is over 190,000). Upstream LLVM builds
# LLVM-C.dll the same way. Listing a symbol also makes the linker pull in the
# object defining it, so the export list, not /WHOLEARCHIVE, decides what the
# DLL contains.
if(WIN32)
  find_program(LLVM_NM_EXECUTABLE
    NAMES llvm-nm
    PATHS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_INSTALL_PREFIX}/bin
    NO_DEFAULT_PATH
  )
  if(NOT LLVM_NM_EXECUTABLE)
    message(FATAL_ERROR
      "llvm-nm not found next to ${LLVM_CONFIG_EXECUTABLE}. It is needed to read "
      "the C API entry points out of LLVM's static archives, which is how the "
      "export list for ${HIP_PYTHON_LIBLLVM_NAME} is built.")
  endif()

  # Target.cpp defines these; they exist nowhere in the archives because LLVM
  # ships them as static inline helpers in llvm-c/Target.h.
  set(LLVM_SHIM_EXPORTS
    LLVMInitializeAllTargetInfos
    LLVMInitializeAllTargets
    LLVMInitializeAllTargetMCs
    LLVMInitializeAllAsmPrinters
    LLVMInitializeAllAsmParsers
    LLVMInitializeAllDisassemblers
    LLVMInitializeNativeTarget
    LLVMInitializeNativeAsmParser
    LLVMInitializeNativeAsmPrinter
    LLVMInitializeNativeDisassembler
  )
  set(LLVM_EXPORTS_EXTRA_ARGS "")
  foreach(_symbol ${LLVM_SHIM_EXPORTS})
    list(APPEND LLVM_EXPORTS_EXTRA_ARGS --extra ${_symbol})
  endforeach()

  set(LLVM_EXPORTS_DEF "${CMAKE_CURRENT_BINARY_DIR}/LLVM.def")
  add_custom_command(
    OUTPUT "${LLVM_EXPORTS_DEF}"
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/gen_msvc_exports.py"
            --nm "${LLVM_NM_EXECUTABLE}"
            --output "${LLVM_EXPORTS_DEF}"
            ${LLVM_EXPORTS_EXTRA_ARGS}
            ${LLVM_LIBFILES_LIST}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_msvc_exports.py" ${LLVM_LIBFILES_LIST}
    COMMENT "Reading the LLVM C API export list out of LLVM's static archives"
    VERBATIM
  )
  add_custom_target(rocmllvm_exports DEPENDS "${LLVM_EXPORTS_DEF}")
endif()

# Create shared library - name it libLLVM.so for easier discovery
add_library(rocmllvm SHARED Target.cpp)
set_target_properties(rocmllvm PROPERTIES OUTPUT_NAME "LLVM")

target_include_directories(rocmllvm
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

# Apply LLVM C++ flags. Not on Windows: llvm-config reports MSVC flags there
# (/EHs-c- /GR-, matching how the archives were built) which contradict the ones
# CMake already puts on the command line, and cl.exe then warns on every
# override. Target.cpp is a C API translation unit that needs neither.
if(NOT WIN32)
  separate_arguments(LLVM_CXXFLAGS_LIST UNIX_COMMAND "${LLVM_CXXFLAGS}")
  target_compile_options(rocmllvm PRIVATE ${LLVM_CXXFLAGS_LIST})
endif()

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
  # No symbol interposition to guard against: DLLs have per-DLL namespaces. The
  # export list generated above takes the place of --whole-archive, and doing
  # both would be counterproductive — /WHOLEARCHIVE would drag every object of
  # every component in, and /OPT:REF would drop them again for want of a
  # reference, leaving the same DLL and a far longer link.
  add_dependencies(rocmllvm rocmllvm_exports)
  set_property(TARGET rocmllvm APPEND PROPERTY LINK_DEPENDS "${LLVM_EXPORTS_DEF}")
  target_link_options(rocmllvm PRIVATE "/DEF:${LLVM_EXPORTS_DEF}")
  target_link_libraries(rocmllvm PRIVATE ${LLVM_LIBFILES_LIST})
endif()

# Apply LLVM linker flags and system libs. NATIVE_COMMAND picks the platform's
# quoting rules: UNIX_COMMAND treats a backslash as an escape, which turns
# llvm-config's Windows output into -LIBPATH:C:Usersrocmlibllvmlib.
separate_arguments(LLVM_LDFLAGS_LIST NATIVE_COMMAND "${LLVM_LDFLAGS}")
separate_arguments(LLVM_SYSTEM_LIBS_LIST NATIVE_COMMAND "${LLVM_SYSTEM_LIBS}")
target_link_options(rocmllvm PRIVATE ${LLVM_LDFLAGS_LIST})
if(EXISTS "${ROCM_SYSDEPS_LIBDIR}")
  if(WIN32)
    target_link_options(rocmllvm PRIVATE "/LIBPATH:${ROCM_SYSDEPS_LIBDIR}")
  else()
    target_link_options(rocmllvm PRIVATE "-L${ROCM_SYSDEPS_LIBDIR}")
  endif()
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
# The Windows import library is an ARCHIVE artifact. Nothing links against the
# bundled library at build time — the bindings load it by name — so it is put in
# a component of its own that no packaging step installs. Left unnamed, CMake
# would drop it into the default lib/ directory and it would surface at the root
# of the wheel.
install(TARGETS rocmllvm
  LIBRARY DESTINATION rocm/bindings/llvm
  RUNTIME DESTINATION rocm/bindings/llvm  # For Windows DLL
  COMPONENT rocm-bindings-compiler
  ARCHIVE DESTINATION rocm/bindings/llvm COMPONENT rocm-bindings-compiler-devel
)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME)

if(BUNDLE_SYSTEM_LIBLLVM)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LIBLLVM_README.txt"
"This directory contains ${HIP_PYTHON_LIBLLVM_NAME} copied from the system ROCm LLVM installation.
Source: ${SYSTEM_LIBLLVM}

This ensures the wheel is self-contained and does not rely on system RPATH.
")
elseif(WIN32)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LIBLLVM_README.txt"
"This directory contains a ${HIP_PYTHON_LIBLLVM_NAME} built from the static LLVM
archives of the ROCm installation at ${LLVM_LIBRARY_DIRS}, because ROCm ships no
shared LLVM on Windows.

It exports the LLVM C API, which is what the bindings call. LLVM's C++ symbols
are not exported: ROCm's archives are compiled without export annotations, and
the full symbol set exceeds what a PE export table can address.
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
