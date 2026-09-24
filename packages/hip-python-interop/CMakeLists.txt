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
# rocm-bindings-core, rocm.bindings.cyhip from rocm-bindings-hip, and
# (for the optional cufile module) rocm.bindings.cyhipfile from
# rocm-bindings-systems.
hip_python_append_sibling_includes(CUDA_INCLUDE_DIRS)

# The cufile interop module (cuda.bindings.cufile / cycufile) is layered on
# hipFILE. It is optional: only build it when the hipfile package is
# installed, mirroring how rocm-bindings-systems gates its hipfile bindings.
find_package(hipfile QUIET)
if(hipfile_FOUND)
  message(STATUS "Found hipfile package — cuda.bindings.cufile interop will be built")
  set(HIP_PYTHON_HIPFILE_AVAILABLE TRUE)
  if(TARGET hip::hipfile)
    get_target_property(HIPFILE_INCLUDE_DIR hip::hipfile INTERFACE_INCLUDE_DIRECTORIES)
  endif()
else()
  message(STATUS "hipfile package not found — cuda.bindings.cufile interop will be skipped")
  set(HIP_PYTHON_HIPFILE_AVAILABLE FALSE)
endif()

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

# Build cuFile bindings (optional — only when hipfile is available)
if(HIP_PYTHON_HIPFILE_AVAILABLE)
  set(CUFILE_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  if(HIPFILE_INCLUDE_DIR)
    list(APPEND CUFILE_INCLUDE_DIRS ${HIPFILE_INCLUDE_DIR})
  endif()

  hip_python_add_cython_module(
    TARGET cuda_bindings_cufile
    MODULE_NAME cuda.bindings.cufile
    SOURCE src/cuda/bindings/cufile.pyx
    DESTINATION cuda/bindings
    COMPONENT hip-python-interop
    CYTHON_INCLUDE_DIRS ${CUFILE_INCLUDE_DIRS}
    INCLUDE_DIRS ${CUFILE_INCLUDE_DIRS}
    DEPENDS ${CUDA_CYTHON_DEPENDS}
  )
endif()

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

# Install the hand-written cufile .pxd files (cimport-only). Gated on hipfile
# availability so the wheel does not advertise cimports for a module it did
# not build.
if(HIP_PYTHON_HIPFILE_AVAILABLE)
  install(
    FILES
      src/cuda/bindings/cufile.pxd
      src/cuda/bindings/cycufile.pxd
    DESTINATION cuda/bindings
    COMPONENT hip-python-interop
  )
endif()

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

# PEP 561 markers for the `cuda` and `cuda.bindings` namespace packages.
# Without them type checkers ignore the .pyi files installed above, so a
# port from cuda-bindings, which does ship types, would lose them. nvtx and
# pynvml stay unmarked: those shims emulate their originals only in part.
install(
  FILES src/cuda/py.typed
  DESTINATION cuda
  COMPONENT hip-python-interop
)
install(
  FILES src/cuda/bindings/py.typed
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
)

# NOTE: cuda/ and cuda/bindings/ are PEP 420 implicit namespace packages
# They intentionally have NO __init__.py files to allow multiple packages
# to contribute modules to the same namespace.
# See: https://peps.python.org/pep-0420/
# Aggregate target for this package (for build.targets in pyproject.toml)
set(_hip_python_interop_targets
  cuda_bindings_driver cuda_bindings_runtime cuda_bindings_nvrtc)
if(HIP_PYTHON_HIPFILE_AVAILABLE)
  list(APPEND _hip_python_interop_targets cuda_bindings_cufile)
endif()
add_custom_target(package_hip_python_interop
  DEPENDS ${_hip_python_interop_targets}
)
set(HIP_PYTHON_INTEROP_MODULE_TARGETS "${_hip_python_interop_targets}" PARENT_SCOPE)
