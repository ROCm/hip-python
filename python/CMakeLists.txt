cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(hip-python-all LANGUAGES C)

# Shared CMake helper
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(HipPythonBuild)

hip_python_initialize()

# Version management: Generate VERSION file from VERSION.in
# HIP_PYTHON_VERSION_SHORT can be set via environment or defaults to commit count
if(NOT DEFINED HIP_PYTHON_VERSION_SHORT)
  execute_process(
    COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    OUTPUT_VARIABLE HIP_PYTHON_VERSION_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT HIP_PYTHON_VERSION_SHORT)
    set(HIP_PYTHON_VERSION_SHORT "0")
  endif()
endif()

# Configure VERSION file at repository root
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION"
  @ONLY
)

# Read the generated VERSION for use in subprojects
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" HIP_PYTHON_VERSION_FULL)
string(STRIP "${HIP_PYTHON_VERSION_FULL}" HIP_PYTHON_VERSION_FULL)

# Set version variables for _version.py.in templates
set(HIP_PYTHON_VERSION_NAME "${HIP_PYTHON_VERSION_FULL}")
set(HIP_PYTHON_LONG_VERSION_NAME "${HIP_PYTHON_VERSION_FULL}")

message(STATUS "hip-python version: ${HIP_PYTHON_VERSION_FULL}")

# Wheel output directory (can be overridden)
# Support DIST_DIR environment variable for backward compatibility
if(DEFINED ENV{DIST_DIR} AND NOT DEFINED HIP_PYTHON_WHEEL_OUTPUT_DIR)
  set(HIP_PYTHON_WHEEL_OUTPUT_DIR "$ENV{DIST_DIR}" CACHE PATH
      "Directory for generated wheel files")
  message(STATUS "Using DIST_DIR from environment: $ENV{DIST_DIR}")
else()
  set(HIP_PYTHON_WHEEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/dist" CACHE PATH
      "Directory for generated wheel files")
endif()

# Auditwheel repair option
option(HIP_PYTHON_AUDITWHEEL_REPAIR "Run auditwheel repair on generated wheels" OFF)

# Early validation: check if auditwheel is available when repair is requested
if(HIP_PYTHON_AUDITWHEEL_REPAIR)
  find_program(AUDITWHEEL_EXECUTABLE auditwheel)
  if(NOT AUDITWHEEL_EXECUTABLE)
    message(FATAL_ERROR
      "HIP_PYTHON_AUDITWHEEL_REPAIR is enabled but auditwheel is not found. "
      "Please install auditwheel (pip install auditwheel) or disable this option.")
  endif()
  message(STATUS "Auditwheel found: ${AUDITWHEEL_EXECUTABLE}")
  message(STATUS "Wheel repair enabled - manylinux wheels will be created")
endif()

# Package build options - allow selective building of packages
option(HIP_PYTHON_BUILD_UTIL "Build rocm-bindings-util package" ON)
option(HIP_PYTHON_BUILD_HIP "Build rocm-bindings-hip package" ON)
option(HIP_PYTHON_BUILD_LIBRARIES "Build rocm-bindings-libraries package" ON)
option(HIP_PYTHON_BUILD_COMPILER "Build rocm-bindings-compiler package" ON)
option(HIP_PYTHON_BUILD_INTEROP "Build hip-python-interop package" ON)

# Control libLLVM.so bundling in rocm-bindings-compiler
option(HIP_PYTHON_BUNDLE_LIBLLVM "Bundle libLLVM.so in rocm-bindings-compiler wheel" ON)
option(HIP_PYTHON_FORCE_BUILD_LIBLLVM
  "Always build libLLVM.so from LLVM static archives (--whole-archive) instead of\
 reusing the system shared library. Implies HIP_PYTHON_BUNDLE_LIBLLVM=ON."
  OFF)

# Note: hip-python is a pure Python metadata package with no CMake build.
# This option only controls whether the hip_python_wheel target is created.
option(HIP_PYTHON_BUILD_HIP_PYTHON "Package hip-python metadata package" ON)

if(NOT HIP_PYTHON_BUILD_HIP_PYTHON)
  message(STATUS "hip-python metadata package disabled (packaging will be skipped)")
endif()

# Validate dependencies
if(HIP_PYTHON_BUILD_HIP AND NOT HIP_PYTHON_BUILD_UTIL)
  message(WARNING "HIP_PYTHON_BUILD_HIP requires HIP_PYTHON_BUILD_UTIL, enabling it")
  set(HIP_PYTHON_BUILD_UTIL ON CACHE BOOL "Build rocm-bindings-util package" FORCE)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES AND NOT HIP_PYTHON_BUILD_UTIL)
  message(WARNING "HIP_PYTHON_BUILD_LIBRARIES requires HIP_PYTHON_BUILD_UTIL, enabling it")
  set(HIP_PYTHON_BUILD_UTIL ON CACHE BOOL "Build rocm-bindings-util package" FORCE)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES AND NOT HIP_PYTHON_BUILD_HIP)
  message(WARNING "HIP_PYTHON_BUILD_LIBRARIES requires HIP_PYTHON_BUILD_HIP, enabling it")
  set(HIP_PYTHON_BUILD_HIP ON CACHE BOOL "Build rocm-bindings-hip package" FORCE)
endif()

if(HIP_PYTHON_BUILD_COMPILER AND NOT HIP_PYTHON_BUILD_UTIL)
  message(WARNING "HIP_PYTHON_BUILD_COMPILER requires HIP_PYTHON_BUILD_UTIL=ON (rocm-bindings-compiler depends on rocm.bindings.util)")
  set(HIP_PYTHON_BUILD_UTIL ON CACHE BOOL "Build rocm-bindings-util package" FORCE)
endif()

# Global include directories for cross-package cimport
# Each path should be the package root so that cimport can resolve
# the full module path (e.g., "from rocm.bindings.cyhip cimport *")
set(HIP_PYTHON_GLOBAL_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(HIP_PYTHON_BUILD_UTIL)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-util"
  )
endif()

if(HIP_PYTHON_BUILD_HIP)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip"
  )
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries"
  )
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler"
  )
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop"
  )
endif()

# Build packages in dependency order (only if enabled)
if(HIP_PYTHON_BUILD_UTIL)
  message(STATUS "Building rocm-bindings-util")
  add_subdirectory(rocm-bindings-util)      # No dependencies
endif()

if(HIP_PYTHON_BUILD_HIP)
  message(STATUS "Building rocm-bindings-hip")
  add_subdirectory(rocm-bindings-hip)       # Depends on util (cimport + import)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  message(STATUS "Building rocm-bindings-libraries")
  add_subdirectory(rocm-bindings-libraries) # Depends on util, hip (cimport + import)
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  message(STATUS "Building rocm-bindings-compiler")
  add_subdirectory(rocm-bindings-compiler)  # Depends on util (cimport + import)
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  message(STATUS "Building hip-python-interop")
  add_subdirectory(hip-python-interop)      # CUDA bindings (driver, runtime, nvrtc)
endif()

# Metadata package (no build, pure Python)
# hip-python/ doesn't need CMake - it's just re-exports

# Wheel packaging targets (conditional on build options)
if(HIP_PYTHON_BUILD_UTIL)
  hip_python_add_wheel_target(
    TARGET util_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-util"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    DEPENDS package_rocm_bindings_util
  )
  message(STATUS "Wheel target enabled: util_wheel")
endif()

if(HIP_PYTHON_BUILD_HIP)
  hip_python_add_wheel_target(
    TARGET hip_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    DEPENDS package_rocm_bindings_hip
  )
  message(STATUS "Wheel target enabled: hip_wheel")
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  hip_python_add_wheel_target(
    TARGET libraries_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    DEPENDS package_rocm_bindings_libraries
  )
  message(STATUS "Wheel target enabled: libraries_wheel")
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  hip_python_add_wheel_target(
    TARGET compiler_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    DEPENDS package_rocm_bindings_compiler
  )
  message(STATUS "Wheel target enabled: compiler_wheel")
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  hip_python_add_wheel_target(
    TARGET interop_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    DEPENDS package_hip_python_interop
  )
  message(STATUS "Wheel target enabled: interop_wheel")
endif()

if(HIP_PYTHON_BUILD_HIP_PYTHON)
  # hip-python is pure Python, no C extensions to depend on
  hip_python_add_wheel_target(
    TARGET hip_python_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hip-python"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Wheel target enabled: hip_python_wheel")
endif()

# Aggregate target to build all wheels
set(ALL_WHEEL_TARGETS)
if(HIP_PYTHON_BUILD_UTIL)
  list(APPEND ALL_WHEEL_TARGETS util_wheel)
endif()
if(HIP_PYTHON_BUILD_HIP)
  list(APPEND ALL_WHEEL_TARGETS hip_wheel)
endif()
if(HIP_PYTHON_BUILD_LIBRARIES)
  list(APPEND ALL_WHEEL_TARGETS libraries_wheel)
endif()
if(HIP_PYTHON_BUILD_COMPILER)
  list(APPEND ALL_WHEEL_TARGETS compiler_wheel)
endif()
if(HIP_PYTHON_BUILD_INTEROP)
  list(APPEND ALL_WHEEL_TARGETS interop_wheel)
endif()
if(HIP_PYTHON_BUILD_HIP_PYTHON)
  list(APPEND ALL_WHEEL_TARGETS hip_python_wheel)
endif()

if(ALL_WHEEL_TARGETS)
  add_custom_target(all_wheels
    DEPENDS ${ALL_WHEEL_TARGETS}
  )
  message(STATUS "Aggregate wheel target 'all_wheels' created with: ${ALL_WHEEL_TARGETS}")
endif()

# ---------------------------------------------------------------------------
# Documentation build (Sphinx)
# ---------------------------------------------------------------------------
#
# The `docs` target is independent of `all_wheels` — it can be built
# standalone, in parallel with the wheel targets, or skipped entirely.
# Sphinx parses Python (and .pyi stub) source files via sphinx-autoapi,
# so the docs build does NOT require the wheels to be installed.
#
option(HIP_PYTHON_BUILD_DOCS "Build the Sphinx documentation" OFF)

set(HIP_PYTHON_DOCS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../docs"
    CACHE PATH "Destination directory for the rendered HTML documentation")
set(HIP_PYTHON_DOCS_DOCTREE_DIR "${CMAKE_BINARY_DIR}/docs/_doctrees"
    CACHE PATH "Directory for Sphinx's intermediate doctree cache")

if(HIP_PYTHON_BUILD_DOCS)
  # Fail fast if Sphinx isn't importable.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sphinx"
    RESULT_VARIABLE _sphinx_check
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT _sphinx_check EQUAL 0)
    message(FATAL_ERROR
      "HIP_PYTHON_BUILD_DOCS=ON but `${Python_EXECUTABLE} -m sphinx` "
      "is not available. Install the docs dependencies (see "
      "docs_src/sphinx/requirements.txt) or disable this option.")
  endif()

  set(_docs_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../docs_src")
  message(STATUS "Sphinx docs target enabled")
  message(STATUS "  source:    ${_docs_source_dir}")
  message(STATUS "  output:    ${HIP_PYTHON_DOCS_OUTPUT_DIR}")
  message(STATUS "  doctrees:  ${HIP_PYTHON_DOCS_DOCTREE_DIR}")

  add_custom_target(docs
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${HIP_PYTHON_DOCS_DOCTREE_DIR}"
    COMMAND "${Python_EXECUTABLE}" -m sphinx
            -T -E -b html
            -d "${HIP_PYTHON_DOCS_DOCTREE_DIR}"
            -D language=en
            "${_docs_source_dir}"
            "${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    WORKING_DIRECTORY "${_docs_source_dir}"
    COMMENT "Building Sphinx HTML documentation → ${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    VERBATIM
  )
endif()
