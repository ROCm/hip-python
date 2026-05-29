cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(hip-python-all LANGUAGES C)

# Shared CMake helper
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(HipPythonBuild)

# Docs-only mode: render the Sphinx documentation without setting up
# the wheel build. Skips hip_python_initialize() (the ROCm path check)
# and forces every per-package HIP_PYTHON_BUILD_* option OFF, so cmake
# configure works in environments without /opt/rocm — most notably
# Read the Docs. See ci/docs/build.sh and .readthedocs.yaml.
option(HIP_PYTHON_BUILD_DOCS_ONLY
       "Render only the Sphinx documentation; skip wheel build setup" OFF)

if(HIP_PYTHON_BUILD_DOCS_ONLY)
  set(HIP_PYTHON_BUILD_DOCS       ON  CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_CORE       OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_HIP        OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_LIBRARIES  OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_SYSTEMS    OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_COMPILER   OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_INTEROP    OFF CACHE BOOL "" FORCE)
  set(HIP_PYTHON_BUILD_HIP_PYTHON OFF CACHE BOOL "" FORCE)
else()
  hip_python_initialize()
endif()

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

# Populate per-package VERSION files (gitignored, single source of truth
# is the repo-root VERSION). Each per-package CMakeLists.txt and
# pyproject.toml then reads `./VERSION`, which lets every package build
# standalone (single-package wheel from its own dir) AND lets sdist
# tarballs include their own VERSION so sdist installs work without
# `../../VERSION` access. See share/design/BUILDING.md.
foreach(_hp_pkg
    rocm-bindings-core rocm-bindings-hip
    rocm-bindings-libraries rocm-bindings-systems
    rocm-bindings-compiler
    hip-python-interop hip-python)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION"
    "${CMAKE_CURRENT_SOURCE_DIR}/${_hp_pkg}/VERSION"
    COPYONLY)
  # Mirror the shared cmake helper into each per-package cmake/ dir so
  # the helper is present locally for both standalone single-package
  # builds AND sdist packaging (sdist.include picks it up from the
  # local path). See share/design/BUILDING.md.
  if(NOT _hp_pkg STREQUAL "hip-python")  # pure Python; no cmake helper needed
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/HipPythonBuild.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_hp_pkg}/cmake/HipPythonBuild.cmake"
      COPYONLY)
  endif()
endforeach()

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
option(HIP_PYTHON_BUILD_CORE "Build rocm-bindings-core package" ON)
option(HIP_PYTHON_BUILD_HIP "Build rocm-bindings-hip package" ON)
option(HIP_PYTHON_BUILD_LIBRARIES "Build rocm-bindings-libraries package" ON)
option(HIP_PYTHON_BUILD_SYSTEMS "Build rocm-bindings-systems package (rccl, roctx)" ON)
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
if(HIP_PYTHON_BUILD_HIP AND NOT HIP_PYTHON_BUILD_CORE)
  message(WARNING "HIP_PYTHON_BUILD_HIP requires HIP_PYTHON_BUILD_CORE, enabling it")
  set(HIP_PYTHON_BUILD_CORE ON CACHE BOOL "Build rocm-bindings-core package" FORCE)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES AND NOT HIP_PYTHON_BUILD_CORE)
  message(WARNING "HIP_PYTHON_BUILD_LIBRARIES requires HIP_PYTHON_BUILD_CORE, enabling it")
  set(HIP_PYTHON_BUILD_CORE ON CACHE BOOL "Build rocm-bindings-core package" FORCE)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES AND NOT HIP_PYTHON_BUILD_HIP)
  message(WARNING "HIP_PYTHON_BUILD_LIBRARIES requires HIP_PYTHON_BUILD_HIP, enabling it")
  set(HIP_PYTHON_BUILD_HIP ON CACHE BOOL "Build rocm-bindings-hip package" FORCE)
endif()

if(HIP_PYTHON_BUILD_SYSTEMS AND NOT HIP_PYTHON_BUILD_CORE)
  message(WARNING "HIP_PYTHON_BUILD_SYSTEMS requires HIP_PYTHON_BUILD_CORE, enabling it")
  set(HIP_PYTHON_BUILD_CORE ON CACHE BOOL "Build rocm-bindings-core package" FORCE)
endif()

if(HIP_PYTHON_BUILD_SYSTEMS AND NOT HIP_PYTHON_BUILD_HIP)
  message(WARNING "HIP_PYTHON_BUILD_SYSTEMS requires HIP_PYTHON_BUILD_HIP, enabling it")
  set(HIP_PYTHON_BUILD_HIP ON CACHE BOOL "Build rocm-bindings-hip package" FORCE)
endif()

if(HIP_PYTHON_BUILD_COMPILER AND NOT HIP_PYTHON_BUILD_CORE)
  message(WARNING "HIP_PYTHON_BUILD_COMPILER requires HIP_PYTHON_BUILD_CORE=ON (rocm-bindings-compiler depends on rocm.bindings.util)")
  set(HIP_PYTHON_BUILD_CORE ON CACHE BOOL "Build rocm-bindings-core package" FORCE)
endif()

# Global include directories for cross-package cimport
# Each path should be the package root so that cimport can resolve
# the full module path (e.g., "from rocm.bindings.cyhip cimport *")
set(HIP_PYTHON_GLOBAL_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(HIP_PYTHON_BUILD_CORE)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-core"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-core/src"
  )
endif()

if(HIP_PYTHON_BUILD_HIP)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip/src"
  )
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries/src"
  )
endif()

if(HIP_PYTHON_BUILD_SYSTEMS)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-systems"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-systems/src"
  )
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler/src"
  )
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  list(APPEND HIP_PYTHON_GLOBAL_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop"
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop/src"
  )
endif()

# Build packages in dependency order (only if enabled)
if(HIP_PYTHON_BUILD_CORE)
  message(STATUS "Building rocm-bindings-core")
  add_subdirectory(rocm-bindings-core)      # No dependencies
endif()

if(HIP_PYTHON_BUILD_HIP)
  message(STATUS "Building rocm-bindings-hip")
  add_subdirectory(rocm-bindings-hip)       # Depends on util (cimport + import)
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  message(STATUS "Building rocm-bindings-libraries")
  add_subdirectory(rocm-bindings-libraries) # Depends on core, hip (cimport + import)
endif()

if(HIP_PYTHON_BUILD_SYSTEMS)
  message(STATUS "Building rocm-bindings-systems")
  add_subdirectory(rocm-bindings-systems)   # Depends on core, hip (cimport + import)
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
if(HIP_PYTHON_BUILD_CORE)
  hip_python_add_wheel_target(
    TARGET core_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-core"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT rocm-bindings-core
    DEPENDS package_rocm_bindings_core
  )
  message(STATUS "Wheel target enabled: core_wheel")
endif()

if(HIP_PYTHON_BUILD_HIP)
  hip_python_add_wheel_target(
    TARGET hip_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT rocm-bindings-hip
    DEPENDS package_rocm_bindings_hip
  )
  message(STATUS "Wheel target enabled: hip_wheel")
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  hip_python_add_wheel_target(
    TARGET libraries_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT rocm-bindings-libraries
    DEPENDS package_rocm_bindings_libraries
  )
  message(STATUS "Wheel target enabled: libraries_wheel")
endif()

if(HIP_PYTHON_BUILD_SYSTEMS)
  hip_python_add_wheel_target(
    TARGET systems_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-systems"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT rocm-bindings-systems
    DEPENDS package_rocm_bindings_systems
  )
  message(STATUS "Wheel target enabled: systems_wheel")
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  hip_python_add_wheel_target(
    TARGET compiler_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT rocm-bindings-compiler
    DEPENDS package_rocm_bindings_compiler
  )
  message(STATUS "Wheel target enabled: compiler_wheel")
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  hip_python_add_wheel_target(
    TARGET interop_wheel
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
    COMPONENT hip-python-interop
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
if(HIP_PYTHON_BUILD_CORE)
  list(APPEND ALL_WHEEL_TARGETS core_wheel)
endif()
if(HIP_PYTHON_BUILD_HIP)
  list(APPEND ALL_WHEEL_TARGETS hip_wheel)
endif()
if(HIP_PYTHON_BUILD_LIBRARIES)
  list(APPEND ALL_WHEEL_TARGETS libraries_wheel)
endif()
if(HIP_PYTHON_BUILD_SYSTEMS)
  list(APPEND ALL_WHEEL_TARGETS systems_wheel)
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

# Sdist packaging targets (conditional on the same per-package opt-in
# as the wheel targets above). Each `<name>_sdist` target runs
# `python -m build --sdist --no-isolation` from its package dir and
# drops the resulting tarball into ${HIP_PYTHON_WHEEL_OUTPUT_DIR}.
# No native build is required — the per-package VERSION and shared
# cmake helper are already in place from the configure_file() loop
# above.
if(HIP_PYTHON_BUILD_CORE)
  hip_python_add_sdist_target(
    TARGET core_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-core"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: core_sdist")
endif()

if(HIP_PYTHON_BUILD_HIP)
  hip_python_add_sdist_target(
    TARGET hip_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: hip_sdist")
endif()

if(HIP_PYTHON_BUILD_LIBRARIES)
  hip_python_add_sdist_target(
    TARGET libraries_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-libraries"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: libraries_sdist")
endif()

if(HIP_PYTHON_BUILD_SYSTEMS)
  hip_python_add_sdist_target(
    TARGET systems_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-systems"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: systems_sdist")
endif()

if(HIP_PYTHON_BUILD_COMPILER)
  hip_python_add_sdist_target(
    TARGET compiler_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-compiler"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: compiler_sdist")
endif()

if(HIP_PYTHON_BUILD_INTEROP)
  hip_python_add_sdist_target(
    TARGET interop_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hip-python-interop"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: interop_sdist")
endif()

if(HIP_PYTHON_BUILD_HIP_PYTHON)
  hip_python_add_sdist_target(
    TARGET hip_python_sdist
    PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hip-python"
    OUTPUT_DIR "${HIP_PYTHON_WHEEL_OUTPUT_DIR}"
  )
  message(STATUS "Sdist target enabled: hip_python_sdist")
endif()

# Aggregate target to build all sdists.
set(ALL_SDIST_TARGETS)
if(HIP_PYTHON_BUILD_CORE)
  list(APPEND ALL_SDIST_TARGETS core_sdist)
endif()
if(HIP_PYTHON_BUILD_HIP)
  list(APPEND ALL_SDIST_TARGETS hip_sdist)
endif()
if(HIP_PYTHON_BUILD_LIBRARIES)
  list(APPEND ALL_SDIST_TARGETS libraries_sdist)
endif()
if(HIP_PYTHON_BUILD_SYSTEMS)
  list(APPEND ALL_SDIST_TARGETS systems_sdist)
endif()
if(HIP_PYTHON_BUILD_COMPILER)
  list(APPEND ALL_SDIST_TARGETS compiler_sdist)
endif()
if(HIP_PYTHON_BUILD_INTEROP)
  list(APPEND ALL_SDIST_TARGETS interop_sdist)
endif()
if(HIP_PYTHON_BUILD_HIP_PYTHON)
  list(APPEND ALL_SDIST_TARGETS hip_python_sdist)
endif()

if(ALL_SDIST_TARGETS)
  add_custom_target(all_sdists
    DEPENDS ${ALL_SDIST_TARGETS}
  )
  message(STATUS "Aggregate sdist target 'all_sdists' created with: ${ALL_SDIST_TARGETS}")
endif()

# ---------------------------------------------------------------------------
# Stubgen: developer-only `.pyi` regeneration for handcoded Cython modules
# ---------------------------------------------------------------------------
#
# When a developer edits one of the handcoded Cython modules listed in
# `HIP_PYTHON_STUBGEN_MODULES` below, they re-run `mypy stubgen` to refresh
# the corresponding `.pyi` next to the `.pyx` and commit both files
# together. This is NOT invoked by any wheel/sdist target — end users
# `pip install`ing from sdist or wheel consume the committed `.pyi`
# directly and do NOT need `mypy` installed.
#
# Opt-in via `-DHIP_PYTHON_ENABLE_STUBGEN=ON`. The targets created are:
#   - `<cython-target>_stub` per module
#   - `<pkg>_stubs` aggregate per package (e.g. `core_stubs`, `hip_stubs`)
#   - `all_stubs` aggregate across packages
#
# Adding a new handcoded Cython module to be stubbed: append one line to
# `HIP_PYTHON_STUBGEN_MODULES` below — this is the single source of
# truth across all packages.
#
# See `share/design/BUILDING.md` "Regenerating stubs for handcoded
# Cython modules" for the full developer workflow.
option(HIP_PYTHON_ENABLE_STUBGEN
  "Enable developer-only stubgen targets (requires `mypy` installed)" OFF)

if(HIP_PYTHON_ENABLE_STUBGEN)
  # Locate the `stubgen` CLI script in the same venv as Python_EXECUTABLE.
  # `python -m mypy.stubgen` does not work when mypy is installed as a
  # compiled .so (`No code object available for mypy.stubgen`); the
  # `stubgen` console-script entry point works regardless.
  get_filename_component(_python_bindir "${Python_EXECUTABLE}" DIRECTORY)
  find_program(HIP_PYTHON_STUBGEN_EXECUTABLE
    NAMES stubgen
    HINTS "${_python_bindir}"
    NO_DEFAULT_PATH
  )
  if(NOT HIP_PYTHON_STUBGEN_EXECUTABLE)
    # Fall back to PATH lookup so a system-wide stubgen still works.
    find_program(HIP_PYTHON_STUBGEN_EXECUTABLE NAMES stubgen)
  endif()
  if(NOT HIP_PYTHON_STUBGEN_EXECUTABLE)
    message(FATAL_ERROR
      "HIP_PYTHON_ENABLE_STUBGEN=ON but `stubgen` was not found "
      "alongside ${Python_EXECUTABLE} or on PATH. Install mypy in "
      "your active Python environment (`pip install mypy`) and "
      "re-configure, or disable this option.")
  endif()
  message(STATUS "stubgen executable: ${HIP_PYTHON_STUBGEN_EXECUTABLE}")

  # Single repo-spanning enumeration. Each entry:
  #   "<package-dir>|<dotted-module>|<source-pyi-relative-dir>|<cython-target>"
  #
  # `<source-pyi-relative-dir>` is relative to `packages/`; it's where the
  # `.pyi` lands in the source tree (next to the `.pyx`).
  # `<cython-target>` is the CMake target whose compiled `.so` stubgen
  # introspects.
  set(HIP_PYTHON_STUBGEN_MODULES
    "core|rocm.bindings.util.types|rocm-bindings-core/src/rocm/bindings/util|rocm_bindings_core_types"
    "core|rocm.bindings.util.loader|rocm-bindings-core/src/rocm/bindings/util|rocm_bindings_core_loader"
    "core|rocm.bindings.util.posixloader|rocm-bindings-core/src/rocm/bindings/util|rocm_bindings_core_platform_loader"
    "hip|rocm.bindings._hip_helpers|rocm-bindings-hip/src/rocm/bindings|rocm_bindings_hip_helpers"
    "hip|rocm.bindings._hiprtc_helpers|rocm-bindings-hip/src/rocm/bindings|rocm_bindings_hiprtc_helpers"
  )

  # Bucket per-module stub targets by package shortname; these populate
  # the per-package <pkg>_stubs aggregate targets below.
  set(_core_stub_targets)
  set(_hip_stub_targets)
  set(_libraries_stub_targets)
  set(_systems_stub_targets)
  set(_compiler_stub_targets)
  set(_interop_stub_targets)

  foreach(_entry IN LISTS HIP_PYTHON_STUBGEN_MODULES)
    string(REPLACE "|" ";" _fields "${_entry}")
    list(GET _fields 0 _pkg_short)
    list(GET _fields 1 _module)
    list(GET _fields 2 _pyi_subdir)
    list(GET _fields 3 _cython_target)

    # Skip entries whose owning package isn't enabled in this configure
    # — the cython target won't exist, and add_custom_target DEPENDS on
    # a missing target would fail.
    set(_owning_var "HIP_PYTHON_BUILD_${_pkg_short}")
    string(TOUPPER "${_owning_var}" _owning_var)
    if(NOT ${_owning_var})
      continue()
    endif()

    hip_python_add_stubgen_target(
      MODULE "${_module}"
      CYTHON_TARGET "${_cython_target}"
      SOURCE_PYI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${_pyi_subdir}"
    )
    list(APPEND _${_pkg_short}_stub_targets "${_cython_target}_stub")
  endforeach()

  set(_all_stub_targets)
  foreach(_pkg_short IN ITEMS core hip libraries systems compiler interop)
    set(_pkg_targets "_${_pkg_short}_stub_targets")
    if(${_pkg_targets})
      add_custom_target(${_pkg_short}_stubs DEPENDS ${${_pkg_targets}})
      list(APPEND _all_stub_targets ${_pkg_short}_stubs)
      message(STATUS "Stubgen target enabled: ${_pkg_short}_stubs (${${_pkg_targets}})")
    endif()
  endforeach()

  if(_all_stub_targets)
    add_custom_target(all_stubs DEPENDS ${_all_stub_targets})
    message(STATUS "Aggregate stubgen target 'all_stubs' created with: ${_all_stub_targets}")
  endif()
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

  # Substitute landing-page version metadata. Read the canonical
  # generated_versions.cmake (rocm-bindings-hip's copy — all five
  # per-package copies carry identical HIP_PYTHON_GENERATED_* values
  # per the interfacegen loop). When a per-source REV is empty (the
  # codegen run did not consult that source tree), surface
  # "*not consulted*" so the rendered note stays informative.
  set(_versions_cmake
      "${CMAKE_CURRENT_SOURCE_DIR}/rocm-bindings-hip/cmake/generated_versions.cmake")
  if(EXISTS "${_versions_cmake}")
    include("${_versions_cmake}")
  endif()
  foreach(_var
      HIP_PYTHON_GENERATED_DATE
      HIP_PYTHON_GENERATED_ROCM_VERSION
      HIP_PYTHON_GENERATED_ROCM_LIBRARIES_REV
      HIP_PYTHON_GENERATED_ROCM_SYSTEMS_REV
      HIP_PYTHON_GENERATED_ROCM_LLVM_PROJECT_REV)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
      set(${_var} "unknown")
    endif()
  endforeach()
  # rocm-systems and llvm-project are always consulted during codegen
  # (hiprtc_runtime.h + the clang Python bindings respectively), so an
  # empty rev there means the codegen run didn't pass the corresponding
  # --rocm-{systems,llvm-project}-dir flag — render that as
  # "*hash not recorded*" so the always-used framing in index.rst.in
  # stays accurate. rocm-libraries is recipe-optional, so "*not consulted*"
  # remains the correct phrasing when its rev is empty.
  foreach(_src ROCM_LIBRARIES ROCM_SYSTEMS ROCM_LLVM_PROJECT)
    set(_rev_var "HIP_PYTHON_GENERATED_${_src}_REV")
    set(_disp_var "HIP_PYTHON_GENERATED_${_src}_REV_DISPLAY")
    if("${${_rev_var}}" STREQUAL "" OR "${${_rev_var}}" STREQUAL "unknown")
      if("${_src}" STREQUAL "ROCM_LIBRARIES")
        set(${_disp_var} "*not consulted*")
      else()
        set(${_disp_var} "*hash not recorded*")
      endif()
    else()
      set(${_disp_var} "``${${_rev_var}}``")
    endif()
  endforeach()
  configure_file(
      "${_docs_source_dir}/index.rst.in"
      "${_docs_source_dir}/index.rst"
      @ONLY)

  # Forward CMAKE_BUILD_PARALLEL_LEVEL to sphinx as -j when set,
  # otherwise leave it unspecified (sphinx's own default = single
  # process). Native `cmake --build … --target docs` is unchanged
  # unless the user passes --parallel N or exports the env var.
  set(_sphinx_jobs_flag)
  if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL}
      AND NOT "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" STREQUAL "")
    set(_sphinx_jobs_flag -j "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
  endif()

  add_custom_target(docs
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${HIP_PYTHON_DOCS_DOCTREE_DIR}"
    COMMAND "${Python_EXECUTABLE}" -m sphinx
            -T -E -b html
            ${_sphinx_jobs_flag}
            -d "${HIP_PYTHON_DOCS_DOCTREE_DIR}"
            -D language=en
            "${_docs_source_dir}"
            "${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    WORKING_DIRECTORY "${_docs_source_dir}"
    COMMENT "Building Sphinx HTML documentation → ${HIP_PYTHON_DOCS_OUTPUT_DIR}"
    VERBATIM
  )
endif()
