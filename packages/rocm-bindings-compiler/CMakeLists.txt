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

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(rocm-bindings-compiler LANGUAGES C CXX)

# Helper resolution (shared vs local)
set(HIP_PYTHON_SHARED_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HipPythonBuild.cmake")
set(HIP_PYTHON_LOCAL_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipPythonBuild.cmake")

if(EXISTS "${HIP_PYTHON_LOCAL_HELPER}")
  include("${HIP_PYTHON_LOCAL_HELPER}")
  message(STATUS "Using local HipPythonBuild.cmake")
elseif(EXISTS "${HIP_PYTHON_SHARED_HELPER}")
  include("${HIP_PYTHON_SHARED_HELPER}")
  message(STATUS "Using shared HipPythonBuild.cmake")
else()
  message(FATAL_ERROR
    "Cannot find HipPythonBuild.cmake helper. "
    "Tried:\n"
    "  - ${HIP_PYTHON_LOCAL_HELPER}\n"
    "  - ${HIP_PYTHON_SHARED_HELPER}")
endif()

hip_python_initialize()

hip_python_resolve_version()

# Find ROCm LLVM and AMD Comgr. Both config packages live under the ROCm
# install, whose internal layout varies by distribution — see
# hip_python_add_rocm_cmake_prefixes. LLVM_DIR is set here rather than left to
# find_package because ROCm exposes a broken LLVM redirect stub on Windows; it
# is then reused by bundled/libllvm/CMakeLists.txt.
hip_python_add_rocm_cmake_prefixes()
find_package(amd_comgr REQUIRED)

hip_python_find_rocm_llvm_dir(LLVM_DIR)
if(NOT LLVM_DIR)
  message(FATAL_ERROR
    "rocm-bindings-compiler needs LLVM's CMake package, and the ROCm "
    "installation at ${ROCM_PATH} does not provide one.\n"
    "On Windows the pip rocm-sdk-devel tree ships the LLVM headers and static "
    "archives but leaves lib/llvm/lib/cmake empty; a TheRock tarball has the "
    "package at lib/llvm/lib/cmake/llvm. Point ROCM_PATH at an installation "
    "that includes it, pass -DLLVM_DIR=<dir containing LLVMConfig.cmake>, or "
    "configure with -DHIP_PYTHON_BUILD_COMPILER=OFF to skip this package.")
endif()
find_package(LLVM REQUIRED CONFIG)

# Build libLLVM.so (bundled version, only if system libLLVM.so not found)
add_subdirectory(bundled/libllvm)

# Collect Cython dependencies
hip_python_collect_cython_depends(COMPILER_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/llvm/**/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/llvm/**/*.pyx"
)

# Include directories
set(COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${LLVM_INCLUDE_DIRS}"
  "${amd_comgr_INCLUDE_DIRS}"
)

# Cross-package cimport: rocm.bindings.util.loader from rocm-bindings-core.
hip_python_append_sibling_includes(COMPILER_INCLUDE_DIRS)

# Define module lists for selective building
set(LLVM_C_MODULES
  disassembler lljit lto debuginfo executionengine
  orc analysis disassemblertypes blake3 remarks
  comdat targetmachine orcee error bitwriter bitreader
  types support core datatypes visibility target
  errorhandling irreader linker lljitutils object deprecated
  # Cython-level (cy*) bindings used via cimport from other modules
  cydisassembler cylljit cylto cydebuginfo cyexecutionengine
  cyorc cyanalysis cydisassemblertypes cyblake3 cyremarks
  cycomdat cytargetmachine cyorcee cyerror cybitwriter cybitreader
  cytypes cysupport cycore cydatatypes cyvisibility cytarget
  cyerrorhandling cyirreader cylinker cylljitutils cyobject cydeprecated
)

set(LLVM_C_TRANSFORMS_MODULES
  passbuilder
  cypassbuilder
)

set(LLVM_CONFIG_MODULES
  llvm_config
  cyllvm_config
)

set(COMGR_MODULES
  comgr
  cycomgr
)

# Build LLVM C modules
foreach(module ${LLVM_C_MODULES})
  # Conditionally link against rocmllvm if we're building it
  if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_c_${module}
      MODULE_NAME rocm.bindings.llvm.c.${module}
      SOURCE src/rocm/bindings/llvm/c/${module}.pyx
      DESTINATION rocm/bindings/llvm/c
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      LINK_LIBRARIES rocmllvm
      DEPENDS ${COMPILER_CYTHON_DEPENDS} rocmllvm
    )
  else()
    # Using system library - no link dependency needed
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_c_${module}
      MODULE_NAME rocm.bindings.llvm.c.${module}
      SOURCE src/rocm/bindings/llvm/c/${module}.pyx
      DESTINATION rocm/bindings/llvm/c
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      DEPENDS ${COMPILER_CYTHON_DEPENDS}
    )
  endif()
  # Set RPATH so module can find libLLVM.so in parent directory
  set_target_properties(rocm_bindings_llvm_c_${module} PROPERTIES
    INSTALL_RPATH "$ORIGIN/.."
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/src"
  )
  list(APPEND LLVM_TARGETS rocm_bindings_llvm_c_${module})
endforeach()

# Build LLVM C transforms modules
foreach(module ${LLVM_C_TRANSFORMS_MODULES})
  # Conditionally link against rocmllvm if we're building it
  if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_c_transforms_${module}
      MODULE_NAME rocm.bindings.llvm.c.transforms.${module}
      SOURCE src/rocm/bindings/llvm/c/transforms/${module}.pyx
      DESTINATION rocm/bindings/llvm/c/transforms
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      LINK_LIBRARIES rocmllvm
      DEPENDS ${COMPILER_CYTHON_DEPENDS} rocmllvm
    )
  else()
    # Using system library - no link dependency needed
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_c_transforms_${module}
      MODULE_NAME rocm.bindings.llvm.c.transforms.${module}
      SOURCE src/rocm/bindings/llvm/c/transforms/${module}.pyx
      DESTINATION rocm/bindings/llvm/c/transforms
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      DEPENDS ${COMPILER_CYTHON_DEPENDS}
    )
  endif()
  # Set RPATH so module can find libLLVM.so (two levels up)
  set_target_properties(rocm_bindings_llvm_c_transforms_${module} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../.."
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/src"
  )
  list(APPEND LLVM_TARGETS rocm_bindings_llvm_c_transforms_${module})
endforeach()

# Build LLVM config modules.
#
# rocm.bindings.llvm.config exposes llvm-config.h's build-configuration macros,
# whose defined-ness varies by platform, so the generated binding references
# some the local LLVM does not define. Reconcile the two before compiling.
hip_python_absent_macro_definitions(_llvm_config_defs
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/llvm/config/cyllvm_config.pxd"
  "${LLVM_INCLUDE_DIRS}/llvm/Config/llvm-config.h")
if(_llvm_config_defs)
  list(JOIN _llvm_config_defs ", " _llvm_config_defs_pretty)
  message(STATUS "llvm-config.h leaves these undefined, supplying them to rocm.bindings.llvm.config: ${_llvm_config_defs_pretty}")
endif()

foreach(module ${LLVM_CONFIG_MODULES})
  # Conditionally link against rocmllvm if we're building it
  if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_config_${module}
      MODULE_NAME rocm.bindings.llvm.config.${module}
      SOURCE src/rocm/bindings/llvm/config/${module}.pyx
      DESTINATION rocm/bindings/llvm/config
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      COMPILE_DEFINITIONS ${_llvm_config_defs}
      LINK_LIBRARIES rocmllvm
      DEPENDS ${COMPILER_CYTHON_DEPENDS} rocmllvm
    )
  else()
    # Using system library - no link dependency needed
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_config_${module}
      MODULE_NAME rocm.bindings.llvm.config.${module}
      SOURCE src/rocm/bindings/llvm/config/${module}.pyx
      DESTINATION rocm/bindings/llvm/config
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      COMPILE_DEFINITIONS ${_llvm_config_defs}
      DEPENDS ${COMPILER_CYTHON_DEPENDS}
    )
  endif()
  # Set RPATH so module can find libLLVM.so in parent directory
  set_target_properties(rocm_bindings_llvm_config_${module} PROPERTIES
    INSTALL_RPATH "$ORIGIN/.."
    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/src"
  )
  list(APPEND LLVM_TARGETS rocm_bindings_llvm_config_${module})
endforeach()

# Build AMD Comgr modules
hip_python_add_cython_module(
  TARGET rocm_bindings_amd_comgr
  MODULE_NAME rocm.bindings.amd_comgr
  SOURCE src/rocm/bindings/amd_comgr.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
  CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  DEPENDS ${COMPILER_CYTHON_DEPENDS}
)

hip_python_add_cython_module(
  TARGET rocm_bindings_cyamd_comgr
  MODULE_NAME rocm.bindings.cyamd_comgr
  SOURCE src/rocm/bindings/cyamd_comgr.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
  CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  DEPENDS ${COMPILER_CYTHON_DEPENDS}
)

list(APPEND COMGR_TARGETS rocm_bindings_amd_comgr rocm_bindings_cyamd_comgr)

# libLLVM.so installation is handled in bundled/libllvm/CMakeLists.txt

# Install the llvm subtree's .pxd (for cimport) and .pyi (type stub) files,
# including the generated __init__.pxd / __init__.pyi namespace markers.
#
# Ownership: rocm/__init__.pxd and rocm/bindings/__init__.pxd are owned and
# installed by rocm-bindings-core; every subfolder a package provides under
# rocm/ (here the rocm/bindings/llvm/** tree) is owned and installed by that
# package — so this wheel installs its own llvm namespace markers.
#
# A single recursive install(DIRECTORY ...) is used rather than per-subfolder
# rules: the llvm C tree is codegen-driven and its subfolders depend on the
# ROCm/LLVM input (they can appear/disappear between versions). install(DIRECTORY)
# recurses and preserves the structure, so it adapts to whatever codegen emits
# and picks up the __init__ markers automatically (no EXCLUDE).
install(
  DIRECTORY src/rocm/bindings/llvm/
  DESTINATION rocm/bindings/llvm
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pxd"
    PATTERN "*.pyi"
)

# Install .pxd files for amd_comgr bindings
install(
  FILES
    src/rocm/bindings/amd_comgr.pxd
    src/rocm/bindings/cyamd_comgr.pxd
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
)

# Install generator-emitted .pyi type stubs for the high-level Python API
# modules at the top of rocm/bindings (e.g. amd_comgr.pyi). cy* modules are
# cimport-only and deliberately not stubbed (see plan §B.2). The llvm subtree's
# .pyi stubs (including __init__.pyi markers) are installed by the recursive
# llvm rule above. This glob is a no-op until interfacegen emits the stubs.
file(GLOB _compiler_pyi_top
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/*.pyi")
if(_compiler_pyi_top)
  install(
    FILES ${_compiler_pyi_top}
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-compiler
  )
endif()

# The `rocm.bindings.clang` shim is upstream LLVM's clang Python bindings with
# two edits: the `clang.enumerations` import is repointed into this package, and
# a libclang resolver fallback is appended to `cindex.py`.
#
# ci/internal/generate-bindings.sh writes them into the tree it commits, so a
# generated branch carries them; a build that generates its own bindings finds
# nothing but the directory's .gitkeep. Stage them here from the same
# llvm-project checkout the codegen reads, so both routes ship the same shim.
#
# The llvm subpackages are PEP 420 namespace packages (markers installed by the
# recursive llvm rule above), so no __init__.py is shipped for them.
set(_hp_clang_upstream
    "${HIP_PYTHON_ROCM_LLVM_PROJECT_DIR}/clang/bindings/python/clang")
set(_hp_clang_staged "${CMAKE_CURRENT_BINARY_DIR}/clang_shim")
set(_hp_clang_intree "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/clang")

if(HIP_PYTHON_ROCM_LLVM_PROJECT_DIR AND EXISTS "${_hp_clang_upstream}")
  message(STATUS "Staging the rocm.bindings.clang shim from ${_hp_clang_upstream}")

  # Restaged from scratch on every configure: the resolver below is appended,
  # so a surviving copy would carry it twice.
  file(REMOVE_RECURSE "${_hp_clang_staged}")
  file(MAKE_DIRECTORY "${_hp_clang_staged}")

  # Whatever the checkout carries rather than a fixed list: enumerations.py,
  # for one, exists only up to ROCm 6.x.
  file(GLOB _hp_clang_files "${_hp_clang_upstream}/*")
  file(COPY ${_hp_clang_files} DESTINATION "${_hp_clang_staged}")
  if(EXISTS "${HIP_PYTHON_ROCM_LLVM_PROJECT_DIR}/LICENSE.TXT")
    file(COPY "${HIP_PYTHON_ROCM_LLVM_PROJECT_DIR}/LICENSE.TXT"
         DESTINATION "${_hp_clang_staged}")
  endif()

  if(NOT EXISTS "${_hp_clang_staged}/cindex.py")
    message(FATAL_ERROR
      "${_hp_clang_upstream} carries no cindex.py; rocm.bindings.clang cannot "
      "be assembled from this llvm-project checkout.")
  endif()

  # enumerations moves with the module, and Config.get_filename has to resolve
  # libclang out of a pip-only ROCm install; see the fragment for why it is
  # appended rather than edited in place.
  file(READ "${_hp_clang_staged}/cindex.py" _hp_cindex)
  string(REPLACE "clang.enumerations" "rocm.bindings.clang.enumerations"
         _hp_cindex "${_hp_cindex}")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libclang_resolver_fallback.py.in"
       _hp_clang_resolver)
  file(WRITE "${_hp_clang_staged}/cindex.py" "${_hp_cindex}${_hp_clang_resolver}")

  install(
    DIRECTORY "${_hp_clang_staged}/"
    DESTINATION rocm/bindings/clang
    COMPONENT rocm-bindings-compiler
  )
elseif(EXISTS "${_hp_clang_intree}/cindex.py")
  # A generated branch. Its cindex.py already carries both edits, so it is
  # installed as it stands.
  message(STATUS "Installing the in-tree rocm.bindings.clang shim")
  install(
    DIRECTORY "${_hp_clang_intree}/"
    DESTINATION rocm/bindings/clang
    COMPONENT rocm-bindings-compiler
    FILES_MATCHING
      PATTERN "*.py"
      PATTERN "LICENSE.TXT"
  )
else()
  message(FATAL_ERROR
    "rocm.bindings.clang has no source: ${_hp_clang_intree} holds no cindex.py "
    "and HIP_PYTHON_ROCM_LLVM_PROJECT_DIR names no llvm-project checkout to "
    "stage it from. Pass -DHIP_PYTHON_ROCM_LLVM_PROJECT_DIR=<llvm-project>.")
endif()

# Install high-level comgr package
install(
  FILES
    src/rocm/comgr/__init__.py
    src/rocm/comgr/comgr.py
    src/rocm/comgr/amd_hsa_kernel_descriptor.py
    src/rocm/comgr/amdhsa_kernel_directives.py
    src/rocm/comgr/hiprtc_header.py
  DESTINATION rocm/comgr
  COMPONENT rocm-bindings-compiler
)
# Aggregate target
set(HIP_PYTHON_COMPILER_MODULE_TARGETS ${LLVM_TARGETS} ${COMGR_TARGETS})
if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
  list(APPEND HIP_PYTHON_COMPILER_MODULE_TARGETS rocmllvm)
  add_custom_target(package_rocm_bindings_compiler
    DEPENDS ${HIP_PYTHON_COMPILER_MODULE_TARGETS}
  )
else()
  add_custom_target(package_rocm_bindings_compiler
    DEPENDS ${HIP_PYTHON_COMPILER_MODULE_TARGETS}
  )
endif()
set(HIP_PYTHON_COMPILER_MODULE_TARGETS "${HIP_PYTHON_COMPILER_MODULE_TARGETS}" PARENT_SCOPE)
