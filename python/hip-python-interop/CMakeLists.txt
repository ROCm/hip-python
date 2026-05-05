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
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda/bindings/*.pyx"
)

set(CUDA_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Use global include dirs if available (unified build)
if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND CUDA_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

# Build CUDA driver bindings
hip_python_add_cython_module(
  TARGET cuda_bindings_driver
  MODULE_NAME cuda.bindings.driver
  SOURCE cuda/bindings/driver.pyx
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
  SOURCE cuda/bindings/runtime.pyx
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
  SOURCE cuda/bindings/nvrtc.pyx
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
  CYTHON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
  DEPENDS ${CUDA_CYTHON_DEPENDS}
)

# Install .pxd files for cimport from other packages
install(
  FILES
    cuda/bindings/driver.pxd
    cuda/bindings/cydriver.pxd
    cuda/bindings/runtime.pxd
    cuda/bindings/cyruntime.pxd
    cuda/bindings/nvrtc.pxd
    cuda/bindings/cynvrtc.pxd
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
)

# Install generator-emitted .pyi type stubs for the high-level Python API
# modules (driver.pyi, runtime.pyi, nvrtc.pyi). cy* modules are
# cimport-only and deliberately not stubbed (see plan §B.2). No-op until
# the interfacegen code generator emits the stubs.
file(GLOB _interop_pyi
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/bindings/*.pyi")
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

# Configure and install version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION cuda/bindings
  COMPONENT hip-python-interop
)

# Aggregate target for this package (for build.targets in pyproject.toml)
add_custom_target(package_hip_python_interop
  DEPENDS cuda_bindings_driver cuda_bindings_runtime cuda_bindings_nvrtc
)
