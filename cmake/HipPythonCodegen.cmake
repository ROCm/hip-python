include_guard(GLOBAL)

# Optional, configure-time HIP Python code generation.
#
# When HIP_PYTHON_RUN_CODEGEN=ON the `hip-python-generate` tool (the
# console script shipped by tools/hip-python-generate) is run DURING
# CMake configure -- before packages/CMakeLists.txt calls
# add_subdirectory() -- so the generated `.pyx`/`.pxd`/`.pyi` sources
# and the per-package `cmake/generated_modules.cmake` lists exist
# before any compile target is created. This is the only model that
# keeps a single `cmake` + single `cmake --build` correct even when the
# generated module set changes (CMake cannot reconfigure mid-build or
# add targets to an already-loaded build graph). See
# share/design/CODEGEN.md.
#
# WARNING: codegen is SLOW. libclang parses every ROCm header; the run
# takes several minutes and can approach ~30 min depending on how many
# cores the parallel workers can use. Because it runs at configure time,
# `cmake -B build` BLOCKS for the whole generation when this is enabled.

option(HIP_PYTHON_RUN_CODEGEN
  "Run the hip-python-generate codegen at configure time (SLOW: several minutes up to ~30 min depending on core count; blocks configure)"
  OFF)

option(HIP_PYTHON_FORCE_CODEGEN
  "Bypass the codegen stamp guard and force regeneration on the next configure"
  OFF)

# ROCm location/version inputs (only required when codegen is ON).
hip_python_get_rocm_path_default(_hip_python_rocm_path_default)
set(HIP_PYTHON_ROCM_PATH "${_hip_python_rocm_path_default}" CACHE PATH
    "ROCm installation passed to the codegen tool (--rocm-path)")
set(HIP_PYTHON_ROCM_VERSION "" CACHE STRING
    "ROCm version passed to the codegen tool (--rocm-version); required when HIP_PYTHON_RUN_CODEGEN=ON")

# Optional codegen header-source overrides.
set(HIP_PYTHON_ROCM_SYSTEMS_DIR "" CACHE PATH
    "Optional rocm-systems repo root passed to codegen (--rocm-systems-dir)")
set(HIP_PYTHON_ROCM_LIBRARIES_DIR "" CACHE PATH
    "Optional rocm-libraries repo root passed to codegen (--rocm-libraries-dir)")
set(HIP_PYTHON_ROCM_LLVM_PROJECT_DIR "" CACHE PATH
    "Optional llvm-project repo root passed to codegen (--rocm-llvm-project-dir)")
set(HIP_PYTHON_CLANG_RESOURCE_DIR "" CACHE PATH
    "Optional libclang resource dir passed to codegen (--clang-resource-dir)")
set(HIP_PYTHON_CODEGEN_INCLUDE "" CACHE STRING
    "Optional space/;-separated subset of wheels to generate (--include): hip libraries systems compiler")

# Declared above the early return below because the packages read it too: a
# library the codegen skipped has no sources to compile, so their module loops
# drop it rather than failing on a target with no .pyx.
option(HIP_PYTHON_CODEGEN_ALLOW_MISSING_HEADERS
  "Skip a library whose header the ROCm at hand does not carry, instead of failing the codegen and the build"
  OFF)
# Comma-separated, not ;-separated: a semicolon would make cmake treat the
# value as a list and expand it into several arguments on the way to the
# codegen, which takes the whole list as one option.
set(HIP_PYTHON_CODEGEN_SKIP_LIBRARIES "" CACHE STRING
    "Optional comma-separated libraries not to generate by name (--skip-libraries), e.g. hiptensor,hipdnn_backend")

if(NOT HIP_PYTHON_RUN_CODEGEN)
  return()
endif()

# --- codegen is enabled from here on ------------------------------------

if(HIP_PYTHON_ROCM_VERSION STREQUAL "")
  message(FATAL_ERROR
    "HIP_PYTHON_RUN_CODEGEN=ON requires -DHIP_PYTHON_ROCM_VERSION=<x.y.z> "
    "(passed to hip-python-generate --rocm-version).")
endif()

# Locate the console script. find_program resolves the platform
# executable (.exe on Windows) and honors the active venv's bin/Scripts.
find_program(HIP_PYTHON_GENERATE_EXECUTABLE hip-python-generate)
if(NOT HIP_PYTHON_GENERATE_EXECUTABLE)
  message(FATAL_ERROR
    "HIP_PYTHON_RUN_CODEGEN=ON but 'hip-python-generate' was not found on "
    "PATH. Install the codegen tool into the active Python environment:\n"
    "  pip install tools/hip-python-generate\n"
    "(its interfacegen dependency is a path-relative editable install; see "
    "tools/hip-python-generate/dev-requirements.txt).")
endif()

# Repo root: packages/.. (the codegen writes into <root>/packages/...).
get_filename_component(_hip_python_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Assemble the optional flags.
set(_hip_python_codegen_optional_flags)
if(NOT HIP_PYTHON_ROCM_SYSTEMS_DIR STREQUAL "")
  list(APPEND _hip_python_codegen_optional_flags --rocm-systems-dir "${HIP_PYTHON_ROCM_SYSTEMS_DIR}")
endif()
if(NOT HIP_PYTHON_ROCM_LIBRARIES_DIR STREQUAL "")
  list(APPEND _hip_python_codegen_optional_flags --rocm-libraries-dir "${HIP_PYTHON_ROCM_LIBRARIES_DIR}")
endif()
if(NOT HIP_PYTHON_ROCM_LLVM_PROJECT_DIR STREQUAL "")
  list(APPEND _hip_python_codegen_optional_flags --rocm-llvm-project-dir "${HIP_PYTHON_ROCM_LLVM_PROJECT_DIR}")
endif()
if(NOT HIP_PYTHON_CLANG_RESOURCE_DIR STREQUAL "")
  list(APPEND _hip_python_codegen_optional_flags --clang-resource-dir "${HIP_PYTHON_CLANG_RESOURCE_DIR}")
endif()
if(NOT HIP_PYTHON_CODEGEN_INCLUDE STREQUAL "")
  string(REPLACE ";" " " _hip_python_codegen_include "${HIP_PYTHON_CODEGEN_INCLUDE}")
  separate_arguments(_hip_python_codegen_include_list NATIVE_COMMAND "${_hip_python_codegen_include}")
  list(APPEND _hip_python_codegen_optional_flags --include ${_hip_python_codegen_include_list})
endif()
if(HIP_PYTHON_CODEGEN_ALLOW_MISSING_HEADERS)
  list(APPEND _hip_python_codegen_optional_flags --allow-missing-headers)
endif()
# Quoted, and only when non-empty: an empty value would append a bare
# --skip-libraries that swallows whatever flag follows it.
if(NOT "${HIP_PYTHON_CODEGEN_SKIP_LIBRARIES}" STREQUAL "")
  list(APPEND _hip_python_codegen_optional_flags
       --skip-libraries "${HIP_PYTHON_CODEGEN_SKIP_LIBRARIES}")
endif()

# Stamp guard: only (re)run when forced, when the input signature
# changed, or when a generated module list is missing. libclang parsing
# is expensive, so we avoid re-running on every no-op reconfigure.
set(_hip_python_codegen_signature
    "v1|${HIP_PYTHON_ROCM_VERSION}|${HIP_PYTHON_ROCM_PATH}|${HIP_PYTHON_ROCM_SYSTEMS_DIR}|${HIP_PYTHON_ROCM_LIBRARIES_DIR}|${HIP_PYTHON_ROCM_LLVM_PROJECT_DIR}|${HIP_PYTHON_CLANG_RESOURCE_DIR}|${HIP_PYTHON_CODEGEN_INCLUDE}|${HIP_PYTHON_CODEGEN_ALLOW_MISSING_HEADERS}|${HIP_PYTHON_CODEGEN_SKIP_LIBRARIES}")
string(SHA256 _hip_python_codegen_signature_hash "${_hip_python_codegen_signature}")
set(_hip_python_codegen_stamp "${CMAKE_BINARY_DIR}/hip_python_codegen.stamp")

set(_hip_python_codegen_needed TRUE)
if(NOT HIP_PYTHON_FORCE_CODEGEN AND EXISTS "${_hip_python_codegen_stamp}")
  file(READ "${_hip_python_codegen_stamp}" _hip_python_codegen_prev)
  string(STRIP "${_hip_python_codegen_prev}" _hip_python_codegen_prev)
  if(_hip_python_codegen_prev STREQUAL _hip_python_codegen_signature_hash
     AND EXISTS "${_hip_python_repo_root}/packages/rocm-bindings-libraries/cmake/generated_modules.cmake")
    set(_hip_python_codegen_needed FALSE)
  endif()
endif()

if(NOT _hip_python_codegen_needed)
  message(STATUS
    "HIP Python codegen: up to date (stamp matches; pass "
    "-DHIP_PYTHON_FORCE_CODEGEN=ON to force a re-run).")
  return()
endif()

message(STATUS
  "Running HIP Python codegen (configure-time) via ${HIP_PYTHON_GENERATE_EXECUTABLE} - "
  "this may take several minutes up to ~30 min depending on core count; "
  "configure will block until it finishes ...")

execute_process(
  COMMAND "${HIP_PYTHON_GENERATE_EXECUTABLE}"
          "${_hip_python_repo_root}"
          --rocm-version "${HIP_PYTHON_ROCM_VERSION}"
          --rocm-path "${HIP_PYTHON_ROCM_PATH}"
          --license-path "${_hip_python_repo_root}/LICENSE"
          ${_hip_python_codegen_optional_flags}
  WORKING_DIRECTORY "${_hip_python_repo_root}"
  RESULT_VARIABLE _hip_python_codegen_rc)

if(NOT _hip_python_codegen_rc EQUAL 0)
  message(FATAL_ERROR "HIP Python codegen failed (exit ${_hip_python_codegen_rc}).")
endif()

file(WRITE "${_hip_python_codegen_stamp}" "${_hip_python_codegen_signature_hash}\n")
message(STATUS "HIP Python codegen complete.")
