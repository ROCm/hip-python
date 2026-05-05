cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(rocm-bindings-hip LANGUAGES C)

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

hip_python_get_env_default(_hip_python_runtime_linking_default HIP_PYTHON_RUNTIME_LINKING ON)
option(HIP_PYTHON_RUNTIME_LINKING "Whether hip-python should resolve HIP libraries at runtime" ${_hip_python_runtime_linking_default})

# Collect Cython dependencies
hip_python_collect_cython_depends(HIP_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pyx"
)

set(HIP_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Use global include dirs if available (unified build), otherwise use local
if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND HIP_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

# Determine link libraries based on runtime linking option
if(NOT HIP_PYTHON_RUNTIME_LINKING)
  set(_hip_libraries amdhip64)
else()
  set(_hip_libraries)
endif()

# Build helper modules
hip_python_add_cython_module(
  TARGET rocm_bindings_hip_helpers
  MODULE_NAME rocm.bindings._hip_helpers
  SOURCE rocm/bindings/_hip_helpers.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

hip_python_add_cython_module(
  TARGET rocm_bindings_hiprtc_helpers
  MODULE_NAME rocm.bindings._hiprtc_helpers
  SOURCE rocm/bindings/_hiprtc_helpers.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

# Build C-level bindings (cyhip, cyhiprtc)
hip_python_add_cython_module(
  TARGET rocm_bindings_cyhip
  MODULE_NAME rocm.bindings.cyhip
  SOURCE rocm/bindings/cyhip.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

hip_python_add_cython_module(
  TARGET rocm_bindings_cyhiprtc
  MODULE_NAME rocm.bindings.cyhiprtc
  SOURCE rocm/bindings/cyhiprtc.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

# Build high-level bindings (hip, hiprtc) - these install at rocm/bindings/ level
hip_python_add_cython_module(
  TARGET rocm_bindings_hip
  MODULE_NAME rocm.bindings.hip
  SOURCE rocm/bindings/hip.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

hip_python_add_cython_module(
  TARGET rocm_bindings_hiprtc
  MODULE_NAME rocm.bindings.hiprtc
  SOURCE rocm/bindings/hiprtc.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  CYTHON_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  INCLUDE_DIRS ${HIP_INCLUDE_DIRS}
  LINK_LIBRARIES ${_hip_libraries}
  DEPENDS ${HIP_CYTHON_DEPENDS}
)

# Install .pxd files for cimport from other packages
install(
  FILES rocm/bindings/cyhip.pxd rocm/bindings/cyhiprtc.pxd rocm/bindings/hip.pxd rocm/bindings/hiprtc.pxd
        rocm/bindings/_hip_helpers.pxd rocm/bindings/_hiprtc_helpers.pxd
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
)

# Install generator-emitted .pyi type stubs alongside the .so extensions
# so static type checkers (mypy, pyright) and IDEs find them automatically.
# Only the high-level Python modules (hip.pyi, hiprtc.pyi) get stubs —
# the cy* C-level modules are cimport-only (see plan §B.2).
# Glob is best-effort: this is a no-op until the interfacegen code generator
# emits the stubs.
file(GLOB _rocm_bindings_hip_pyi
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pyi")
if(_rocm_bindings_hip_pyi)
  install(
    FILES ${_rocm_bindings_hip_pyi}
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-hip
  )
endif()

# Install Python extension module (hiprtc_pyext.py)
install(
  FILES rocm/bindings/hiprtc_pyext.py
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
)

# Configure and install version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
)

# NOTE: rocm/bindings/ is a PEP 420 implicit namespace package
# It intentionally has NO __init__.py file to allow multiple packages
# (rocm-bindings-core, rocm-bindings-hip, rocm-bindings-libraries)
# to contribute modules to the same namespace.
# See: https://peps.python.org/pep-0420/

# Aggregate target for this package (for build.targets in pyproject.toml)
add_custom_target(package_rocm_bindings_hip
  DEPENDS
    rocm_bindings_hip_helpers
    rocm_bindings_hiprtc_helpers
    rocm_bindings_cyhip
    rocm_bindings_cyhiprtc
    rocm_bindings_hip
    rocm_bindings_hiprtc
)
