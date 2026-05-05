cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(rocm-bindings-core LANGUAGES C)

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
hip_python_collect_cython_depends(UTIL_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/util/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/util/*.pyx"
)

set(UTIL_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Use global include dirs if available (unified build), otherwise use local
if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND UTIL_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

# Build types module
hip_python_add_cython_module(
  TARGET rocm_bindings_core_types
  MODULE_NAME rocm.bindings.util.types
  SOURCE rocm/bindings/util/types.pyx
  DESTINATION rocm/bindings/util
  COMPONENT rocm-bindings-core
  CYTHON_INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
  INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
  DEPENDS ${UTIL_CYTHON_DEPENDS}
)

# Build platform-specific loader (posixloader on Linux, win32loader on Windows)
if(WIN32)
  hip_python_add_cython_module(
    TARGET rocm_bindings_core_platform_loader
    MODULE_NAME rocm.bindings.util.win32loader
    SOURCE rocm/bindings/util/win32loader.pyx
    DESTINATION rocm/bindings/util
    COMPONENT rocm-bindings-core
    CYTHON_INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
    INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
    DEPENDS ${UTIL_CYTHON_DEPENDS}
  )
  target_link_libraries(rocm_bindings_core_platform_loader PRIVATE kernel32)
else()
  hip_python_add_cython_module(
    TARGET rocm_bindings_core_platform_loader
    MODULE_NAME rocm.bindings.util.posixloader
    SOURCE rocm/bindings/util/posixloader.pyx
    DESTINATION rocm/bindings/util
    COMPONENT rocm-bindings-core
    CYTHON_INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
    INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
    DEPENDS ${UTIL_CYTHON_DEPENDS}
  )
endif()

# Build platform-agnostic loader.pyx which re-exports the platform-specific loader
# Uses Cython IF UNAME_SYSNAME to select posixloader (Linux) or win32loader (Windows)
hip_python_add_cython_module(
  TARGET rocm_bindings_core_loader
  MODULE_NAME rocm.bindings.util.loader
  SOURCE rocm/bindings/util/loader.pyx
  DESTINATION rocm/bindings/util
  COMPONENT rocm-bindings-core
  CYTHON_INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
  INCLUDE_DIRS ${UTIL_INCLUDE_DIRS}
  DEPENDS ${UTIL_CYTHON_DEPENDS}
)

# Loader depends on platform-specific loader being built first
add_dependencies(rocm_bindings_core_loader rocm_bindings_core_platform_loader)
if(WIN32)
  target_link_libraries(rocm_bindings_core_loader PRIVATE kernel32)
endif()

# Install .pxd files (for cimport from other packages)
install(
  FILES rocm/bindings/util/loader.pxd
  DESTINATION rocm/bindings/util
  COMPONENT rocm-bindings-core
)

# Install __init__.pxd files for package hierarchy resolution
# IMPORTANT: These MUST be installed for rocm-bindings-core because
# other packages (hip, libraries, compiler) cimport from util
install(
  FILES rocm/__init__.pxd
  DESTINATION rocm
  COMPONENT rocm-bindings-core
)
install(
  FILES rocm/bindings/__init__.pxd
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-core
)

# Install .pxd files for cimport from other packages
if(WIN32)
  install(
    FILES
      rocm/bindings/util/types.pxd
      rocm/bindings/util/win32loader.pxd
    DESTINATION rocm/bindings/util
    COMPONENT rocm-bindings-core
  )
else()
  install(
    FILES
      rocm/bindings/util/types.pxd
      rocm/bindings/util/posixloader.pxd
    DESTINATION rocm/bindings/util
    COMPONENT rocm-bindings-core
  )
endif()

# Configure and install version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION rocm/bindings/util
  COMPONENT rocm-bindings-core
)

# Install Python modules
install(
  FILES
    rocm/bindings/util/__init__.py
    rocm/bindings/util/paths.py
  DESTINATION rocm/bindings/util
  COMPONENT rocm-bindings-core
)

# Aggregate target for this package (for build.targets in pyproject.toml)
add_custom_target(package_rocm_bindings_core
  DEPENDS rocm_bindings_core_types rocm_bindings_core_loader rocm_bindings_core_platform_loader
)
