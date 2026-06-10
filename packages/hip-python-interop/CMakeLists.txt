cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(hip-python-interop LANGUAGES C)

# Helper resolution: try shared helper first (for unified build), then local
set(HIP_PYTHON_SHARED_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HipPythonBuild.cmake")
set(HIP_PYTHON_LOCAL_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipPythonBuild.cmake")

# For sdist: copy shared helper to local cmake/ directory
if(EXISTS "${HIP_PYTHON_SHARED_HELPER}" AND DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "sdist")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  configure_file("${HIP_PYTHON_SHARED_HELPER}" "${HIP_PYTHON_LOCAL_HELPER}" COPYONLY)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(EXISTS "${HIP_PYTHON_SHARED_HELPER}")
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
endif()
include(HipPythonBuild)

hip_python_initialize()

hip_python_resolve_version()

# Collect Cython dependencies
hip_python_collect_cython_depends(CUDA_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/bindings/*.pyx"
)

set(CUDA_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Cross-package cimports: rocm.bindings.util.loader from
# rocm-bindings-core, rocm.bindings.cyhip from rocm-bindings-hip.
hip_python_append_sibling_includes(CUDA_INCLUDE_DIRS)

# Build CUDA driver bindings
hip_python_add_cython_module(
  TARGET cuda_bindings_driver
  MODULE_NAME cuda.bindings.driver
  SOURCE src/cuda/bindings/driver.pyx
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
  CYTHON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  DEPENDS ${CUDA_CYTHON_DEPENDS}
)

# Build CUDA runtime bindings
hip_python_add_cython_module(
  TARGET cuda_bindings_runtime
  MODULE_NAME cuda.bindings.runtime
  SOURCE src/cuda/bindings/runtime.pyx
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
  CYTHON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  DEPENDS ${CUDA_CYTHON_DEPENDS}
)

# Build NVRTC bindings
hip_python_add_cython_module(
  TARGET cuda_bindings_nvrtc
  MODULE_NAME cuda.bindings.nvrtc
  SOURCE src/cuda/bindings/nvrtc.pyx
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
  CYTHON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  DEPENDS ${CUDA_CYTHON_DEPENDS}
)

# Install .pxd files for cimport from other packages
install(
  FILES
    src/cuda/bindings/driver.pxd
    src/cuda/bindings/cydriver.pxd
    src/cuda/bindings/runtime.pxd
    src/cuda/bindings/cyruntime.pxd
    src/cuda/bindings/nvrtc.pxd
    src/cuda/bindings/cynvrtc.pxd
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
)

# Install generator-emitted .pyi type stubs for the high-level Python API
# modules (driver.pyi, runtime.pyi, nvrtc.pyi). cy* modules are
# cimport-only and deliberately not stubbed (see plan §B.2). No-op until
# the interfacegen code generator emits the stubs.
file(GLOB _interop_pyi
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/bindings/*.pyi")
if(_interop_pyi)
  install(
    FILES ${_interop_pyi}
    DESTINATION cuda/bindings
    COMPONENT hip-python-interop
  )
endif()

# NOTE: cuda/ and cuda/bindings/ are PEP 420 implicit namespace packages
# They intentionally have NO __init__.py files to allow multiple packages
# to contribute modules to the same namespace.
# See: https://peps.python.org/pep-0420/
# Aggregate target for this package (for build.targets in pyproject.toml)
add_custom_target(package_hip_python_interop
  DEPENDS cuda_bindings_driver cuda_bindings_runtime cuda_bindings_nvrtc
)
