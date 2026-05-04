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

# Find ROCm LLVM and AMD Comgr
find_package(amd_comgr REQUIRED)

# Find LLVM (will be used by src/CMakeLists.txt)
if(DEFINED ROCM_PATH)
  set(LLVM_DIR "${ROCM_PATH}/llvm/lib/cmake/llvm")
endif()
find_package(LLVM REQUIRED CONFIG)

# Build libLLVM.so (bundled version, only if system libLLVM.so not found)
add_subdirectory(src)

# Collect Cython dependencies
hip_python_collect_cython_depends(COMPILER_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/llvm/**/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/llvm/**/*.pyx"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/comgr/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/comgr/*.pyx"
)

# Include directories
set(COMPILER_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LLVM_INCLUDE_DIRS}"
  "${amd_comgr_INCLUDE_DIRS}"
)

if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND COMPILER_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

# Define module lists for selective building
# Based on ROCM_LLVM_PYTHON_MODULE_LIST from _version.py.in
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
      SOURCE rocm/bindings/llvm/c/${module}.pyx
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
      SOURCE rocm/bindings/llvm/c/${module}.pyx
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
      SOURCE rocm/bindings/llvm/c/transforms/${module}.pyx
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
      SOURCE rocm/bindings/llvm/c/transforms/${module}.pyx
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

# Build LLVM config modules
foreach(module ${LLVM_CONFIG_MODULES})
  # Conditionally link against rocmllvm if we're building it
  if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_config_${module}
      MODULE_NAME rocm.bindings.llvm.config.${module}
      SOURCE rocm/bindings/llvm/config/${module}.pyx
      DESTINATION rocm/bindings/llvm/config
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      LINK_LIBRARIES rocmllvm
      DEPENDS ${COMPILER_CYTHON_DEPENDS} rocmllvm
    )
  else()
    # Using system library - no link dependency needed
    hip_python_add_cython_module(
      TARGET rocm_bindings_llvm_config_${module}
      MODULE_NAME rocm.bindings.llvm.config.${module}
      SOURCE rocm/bindings/llvm/config/${module}.pyx
      DESTINATION rocm/bindings/llvm/config
      COMPONENT rocm-bindings-compiler
      CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
      INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
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
  SOURCE rocm/bindings/amd_comgr.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
  CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  LINK_LIBRARIES amd_comgr
  DEPENDS ${COMPILER_CYTHON_DEPENDS}
)

hip_python_add_cython_module(
  TARGET rocm_bindings_cyamd_comgr
  MODULE_NAME rocm.bindings.cyamd_comgr
  SOURCE rocm/bindings/cyamd_comgr.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
  CYTHON_INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  INCLUDE_DIRS ${COMPILER_INCLUDE_DIRS}
  LINK_LIBRARIES amd_comgr
  DEPENDS ${COMPILER_CYTHON_DEPENDS}
)

list(APPEND COMGR_TARGETS rocm_bindings_amd_comgr rocm_bindings_cyamd_comgr)

# libLLVM.so installation is handled in src/CMakeLists.txt

# Install .pxd files (for cimport)
# NOTE: __init__.pxd files are intentionally excluded — they are build-time
# package markers only. The installed namespace markers come from
# rocm-bindings-util.
install(
  DIRECTORY rocm/bindings/llvm/c/
  DESTINATION rocm/bindings/llvm/c
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pxd"
    PATTERN "__init__.pxd" EXCLUDE
)

install(
  DIRECTORY rocm/bindings/llvm/config/
  DESTINATION rocm/bindings/llvm/config
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pxd"
    PATTERN "__init__.pxd" EXCLUDE
)

# Install .pxd files for amd_comgr bindings
install(
  FILES
    rocm/bindings/amd_comgr.pxd
    rocm/bindings/cyamd_comgr.pxd
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-compiler
)

# Install generator-emitted .pyi type stubs for the high-level Python API
# modules. cy* modules are cimport-only and deliberately not stubbed (see
# plan §B.2). All globs are no-ops until interfacegen actually emits the
# stubs.
file(GLOB _compiler_pyi_top
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pyi")
if(_compiler_pyi_top)
  install(
    FILES ${_compiler_pyi_top}
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-compiler
  )
endif()
install(
  DIRECTORY rocm/bindings/llvm/c/
  DESTINATION rocm/bindings/llvm/c
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pyi"
)
install(
  DIRECTORY rocm/bindings/llvm/config/
  DESTINATION rocm/bindings/llvm/config
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pyi"
)

# Install Python modules
install(
  FILES
    rocm/bindings/llvm/__init__.py
    rocm/bindings/llvm/c/__init__.py
    rocm/bindings/llvm/config/__init__.py
    rocm/bindings/clang/__init__.py
    rocm/bindings/clang/cindex.py
  DESTINATION rocm/bindings/
  COMPONENT rocm-bindings-compiler
)

# Install high-level comgr package
install(
  FILES
    rocm/comgr/__init__.py
    rocm/comgr/comgr.py
    rocm/comgr/amd_hsa_kernel_descriptor.py
    rocm/comgr/amdhsa_kernel_directives.py
    rocm/comgr/hiprtc_runtime.h
  DESTINATION rocm/comgr
  COMPONENT rocm-bindings-compiler
)

# Version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION rocm/bindings/llvm
  COMPONENT rocm-bindings-compiler
)

# Aggregate target
if(TARGET rocmllvm AND BUILD_BUNDLED_LIBLLVM)
  add_custom_target(package_rocm_bindings_compiler
    DEPENDS ${LLVM_TARGETS} ${COMGR_TARGETS} rocmllvm
  )
else()
  add_custom_target(package_rocm_bindings_compiler
    DEPENDS ${LLVM_TARGETS} ${COMGR_TARGETS}
  )
endif()
