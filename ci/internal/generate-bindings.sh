#!/usr/bin/env bash
set -xeu

# Isolated hip-python code generation.
#
# This is the STANDALONE generator run used to produce a release branch's
# generated tree. It is distinct from the cmake configure-time codegen that
# ci/internal/build-wheels.sh can trigger via -DHIP_PYTHON_RUN_CODEGEN=ON;
# this script runs the generator directly and additionally produces the two
# generated artifacts that need ROCm component source/build trees:
#
#   1. The Cython bindings (.pxd/.pyx/.pyi), per-package
#      cmake/generated_{modules,versions}.cmake, and the docs-side outputs
#      (docs_src/sphinx/_toc.yml.in + docs_src/python_api/*.rst) — emitted by
#      `hip-python-generate`.
#   2. The hiprtc runtime header (rocm.comgr/hiprtc_runtime.h) — built from
#      the clr/hipamd `hiprtc-builtins` target.
#   3. The rocm.bindings.clang bindings (cindex.py et al. + LLVM LICENSE.TXT)
#      — copied from the llvm-project clang Python bindings.
#
# The generated tree is written into ${BUILD_DIR}/hip_python so the sibling
# commit step (ci/internal/commit-bindings.sh) can stage and commit it.
#
# Required env:
#   SRC_DIR      parent dir holding hip_python/ and the ROCm component repos
#                (rocm_systems/, rocm_libraries/, rocm_llvm_project/)
#   BUILD_DIR    scratch dir; the working copy lands at ${BUILD_DIR}/hip_python
#   ROCM_VERSION ROCm version passed to the generator (e.g. 7.13.0)
#
# Optional env:
#   ROCM_PATH                   default /opt/rocm
#   ROCM_SYSTEMS_DIR            default ${SRC_DIR}/rocm_systems
#   ROCM_LIBRARIES_DIR          default ${SRC_DIR}/rocm_libraries
#   ROCM_LLVM_PROJECT_DIR       default ${SRC_DIR}/rocm_llvm_project
#   INTERFACEGEN_DIR            default ${SRC_DIR}/hip_python/tools/interfacegen
#   HIP_PYTHON_CODEGEN_DIR      default ${SRC_DIR}/hip_python/tools/hip-python-generate
#   HIP_PYTHON_CODEGEN_BASE_BRANCH
#                               base branch whose generator-owned content is
#                               restored before regeneration so the diff
#                               reflects only the generator's output.
#                               default amd-integration

### resolved paths

src_dir=${SRC_DIR}/hip_python
build_dir=${BUILD_DIR}/hip_python

rocm_path=${ROCM_PATH:-/opt/rocm}
rocm_systems_dir=${ROCM_SYSTEMS_DIR:-${SRC_DIR}/rocm_systems}
rocm_libraries_dir=${ROCM_LIBRARIES_DIR:-${SRC_DIR}/rocm_libraries}
rocm_llvm_project_dir=${ROCM_LLVM_PROJECT_DIR:-${SRC_DIR}/rocm_llvm_project}
interfacegen_dir=${INTERFACEGEN_DIR:-${src_dir}/tools/interfacegen}
hip_python_codegen_dir=${HIP_PYTHON_CODEGEN_DIR:-${src_dir}/tools/hip-python-generate}
base_branch=${HIP_PYTHON_CODEGEN_BASE_BRANCH:-amd-integration}

### prepare an isolated working copy

rm -rf ${build_dir}
mkdir -p ${BUILD_DIR}
cp -R ${src_dir} ${BUILD_DIR}/

cd ${build_dir}

# Clean generator-owned content so the commit step can stage a delta. The
# packages/<wheel>/src/rocm/bindings/*.{pxd,pyx,pyi} and
# cmake/generated_{modules,versions}.cmake files are regenerated below;
# restore them from the base branch first so the diff reflects only the
# generator's output.
for pkg in rocm-bindings-core rocm-bindings-hip rocm-bindings-libraries \
           rocm-bindings-systems rocm-bindings-compiler hip-python-interop; do
  git checkout origin/${base_branch} -- packages/${pkg}/src packages/${pkg}/cmake \
    2>/dev/null || true
done

### step 1 — install the in-tree codegen tooling

# interfacegen + the hip-python-generate recipe live in the hip-python
# monorepo under tools/. libclang Python bindings often lag the system
# libclang.so; pin a known-good range. Installed into the system Python so
# the `hip-python-generate` console script stays on PATH for the run below.
python3 -m pip install --upgrade pip
python3 -m pip install "${interfacegen_dir}"
python3 -m pip install "${hip_python_codegen_dir}"
python3 -m pip install "libclang>=18,<19"

### step 2 — generate the Cython bindings + docs outputs

hip-python-generate ${build_dir} \
  --rocm-version ${ROCM_VERSION} \
  --rocm-path ${rocm_path} \
  --rocm-systems-dir ${rocm_systems_dir} \
  --rocm-libraries-dir ${rocm_libraries_dir} \
  --rocm-llvm-project-dir ${rocm_llvm_project_dir} \
  --license-path ${build_dir}/LICENSE

### step 3 — generate the hiprtc runtime header

cp -R ${rocm_systems_dir}/projects/clr ${BUILD_DIR}/

# create venv in temp directory
VENV_DIR=$(mktemp -d)
python3 -m venv ${VENV_DIR}
. ${VENV_DIR}/bin/activate
python3 -m pip install cmake cppheaderparser

PYTHON3_EXECUTABLE="${VENV_DIR}/bin/python3"

# FIXME(HIP/AMD): add missing cmake_minimum_required(..) to CMakeLists.txt,
# as it is a hard error with recent CMake versions
if ! grep -q "cmake_minimum_required" ${BUILD_DIR}/clr/hipamd/CMakeLists.txt; then
    cp ${BUILD_DIR}/clr/hipamd/CMakeLists.txt ${BUILD_DIR}/clr/hipamd/CMakeLists.txt.orig
    printf 'cmake_minimum_required(VERSION 3.16.8)\n\n' > ${BUILD_DIR}/clr/hipamd/CMakeLists.txt.tmp
    cat ${BUILD_DIR}/clr/hipamd/CMakeLists.txt.orig >> ${BUILD_DIR}/clr/hipamd/CMakeLists.txt.tmp
    mv ${BUILD_DIR}/clr/hipamd/CMakeLists.txt.tmp ${BUILD_DIR}/clr/hipamd/CMakeLists.txt
fi

# create build directory in temp location
HIPAMD_BUILD_DIR=$(mktemp -d)

HIP_DIR=${rocm_systems_dir}/projects/hip
HIPCC_BIN_DIR=${rocm_path}/bin
HIP_LLVM_ROOT=${rocm_path}/llvm
OPENCL_DIR=${rocm_systems_dir}/projects/clr/opencl
ROCCLR_DIR=${rocm_systems_dir}/projects/clr/rocclr
HIP_PLATFORM=amd

cmake -S ${BUILD_DIR}/clr/hipamd \
      -B ${HIPAMD_BUILD_DIR} \
      -DHIP_COMMON_DIR="${HIP_DIR}" \
      -DHIPCC_BIN_DIR="${HIPCC_BIN_DIR}" \
      -DHIP_LLVM_ROOT="${HIP_LLVM_ROOT}" \
      -DAMD_OPENCL_PATH=${OPENCL_DIR} \
      -DROCCLR_PATH=${ROCCLR_DIR} \
      -DCMAKE_PREFIX_PATH="${rocm_path}/" \
      -DPython3_EXECUTABLE="${PYTHON3_EXECUTABLE}" \
      -DCMAKE_INSTALL_PREFIX=install \
      --fresh --debug-output

# generate hiprtc runtime header
echo "generate hiprtc runtime header"
cmake --build ${HIPAMD_BUILD_DIR} --target hiprtc-builtins --verbose

# copy header file into the merged hip-python tree's rocm-bindings-compiler wheel
echo "copy header file into rocm-bindings-compiler 'rocm.comgr' package"
cp ${HIPAMD_BUILD_DIR}/src/hiprtc/hip_rtc_gen/hipRTC \
   ${build_dir}/packages/rocm-bindings-compiler/src/rocm/comgr/hiprtc_runtime.h

deactivate
rm -rf ${VENV_DIR} ${HIPAMD_BUILD_DIR}

### step 4 — copy the rocm.bindings.clang bindings

cd ${build_dir}/packages/rocm-bindings-compiler/src/rocm/bindings/clang/

  # 1) copy clang bindings into the rocm-bindings-compiler wheel's
  #    `rocm.bindings.clang` package.
  echo "copy clang bindings into rocm.bindings.clang package"
  for f in "__init__.py" "cindex.py"; do
      rm -f ${f}
      cp ${rocm_llvm_project_dir}/clang/bindings/python/clang/${f} .
  done
  # optional files, do not exist for all rocm versions
  for f in "enumerations.py"; do
      rm -f ${f}
      cp ${rocm_llvm_project_dir}/clang/bindings/python/clang/${f} . || true
  done
  sed -s -i "s,clang\.enumerations,rocm.bindings.clang.enumerations," \
    ${build_dir}/packages/rocm-bindings-compiler/src/rocm/bindings/clang/cindex.py

  # 2) copy LLVM LICENSE.TXT next to the clang bindings.
  echo "copy LLVM LICENSE.TXT into rocm.bindings.clang package"
  rm -f LICENSE.TXT
  cp ${rocm_llvm_project_dir}/LICENSE.TXT .
