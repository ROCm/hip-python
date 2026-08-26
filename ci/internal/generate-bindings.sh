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
#      (docs_src/sphinx/_toc.yml.in + docs_src/python_api/*.rst) - emitted by
#      `hip-python-generate`.
#   2. The rocm.bindings.clang bindings (cindex.py et al. + LLVM LICENSE.TXT)
#      - copied from the llvm-project clang Python bindings.
#
# The hipRTC runtime header is deliberately not produced here. rocm.comgr reads
# that text out of the installed ROCm's hiprtc-builtins library rather than from
# a copy obtained by configuring clr/hipamd and building its `hiprtc-builtins`
# target, which is both cheaper and correct: a generated file carries the data
# model of whichever host generated it, and a Linux-generated copy is invalid
# input to a Windows HIP compilation.
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
#   HIP_PYTHON_ALLOW_MISSING_HEADERS
#                               'true' passes --allow-missing-headers, which
#                               turns a library whose header this ROCm never
#                               shipped into a skip rather than a failure. For
#                               generating against an older ROCm; leave it off
#                               and a missing header stays an error.
#                               default false
#   HIP_PYTHON_SKIP_LIBRARIES   comma-separated libraries not to generate, by
#                               name, e.g. 'hiptensor,hipdnn_backend'. Passed
#                               on whole as --skip-libraries; an unknown name
#                               fails the run.
#                               default empty

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
allow_missing_headers=${HIP_PYTHON_ALLOW_MISSING_HEADERS:-false}
skip_libraries=${HIP_PYTHON_SKIP_LIBRARIES:-}

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

### step 1 - install the in-tree codegen tooling

# interfacegen + the hip-python-generate recipe live in the hip-python
# monorepo under tools/. libclang Python bindings often lag the system
# libclang.so; pin a known-good range. Installed into the system Python so
# the `hip-python-generate` console script stays on PATH for the run below.
python3 -m pip install --upgrade pip
python3 -m pip install "${interfacegen_dir}"
python3 -m pip install "${hip_python_codegen_dir}"
python3 -m pip install "libclang>=18,<19"

### step 2 - generate the Cython bindings + docs outputs

# Best-effort generation: a single failing library must not abort the whole
# run. Capture the generator's exit status (rather than letting `set -e` bail
# out here) so the remaining steps still run and the partial tree is fully
# populated; the captured status is re-raised at the very end of the script.
generate_args=(
  ${build_dir}
  --rocm-version ${ROCM_VERSION}
  --rocm-path ${rocm_path}
  --rocm-systems-dir ${rocm_systems_dir}
  --rocm-libraries-dir ${rocm_libraries_dir}
  --rocm-llvm-project-dir ${rocm_llvm_project_dir}
  --license-path ${build_dir}/LICENSE
)
if [[ "${allow_missing_headers}" == "true" ]]; then
  generate_args+=(--allow-missing-headers)
fi
# Quoted: the generator takes the comma-separated list as one argument.
if [[ -n "${skip_libraries}" ]]; then
  generate_args+=(--skip-libraries "${skip_libraries}")
fi

gen_rc=0
hip-python-generate "${generate_args[@]}" || gen_rc=$?
if [[ ${gen_rc} -ne 0 ]]; then
  echo "[warn] some libraries failed to generate; continuing to produce the most complete partial tree possible (see the per-library logs above)"
fi

### step 3 - copy the rocm.bindings.clang bindings

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

  # 1b) append the libclang resolver fallback to the copied cindex.py.
  #     Upstream Config.get_filename() returns a bare soname (e.g.
  #     "libclang.so") when neither Config.set_library_file/set_library_path
  #     nor the LIBCLANG_* env vars are set; that fails in a pip-only ROCm
  #     install. Wrap it so the unconfigured, non-absolute (bare-soname) case
  #     resolves libclang through the shared rocm-bindings resolver (ROCM_PATH,
  #     the rocm_sdk wheel anchor, versioned soname). Explicit Config/LIBCLANG_*
  #     overrides already yield an absolute path and pass straight through, so
  #     their priority is preserved. Appending (rather than editing the
  #     get_filename body) keeps this robust to upstream LLVM changes.
  #
  #     The text lives in a file rather than a heredoc because the wheel build
  #     appends the same fragment when it stages the shim itself (see
  #     packages/rocm-bindings-compiler/CMakeLists.txt); one copy cannot drift
  #     from the other.
  echo "append libclang resolver fallback to rocm.bindings.clang.cindex"
  cat ${build_dir}/packages/rocm-bindings-compiler/cmake/libclang_resolver_fallback.py.in \
    >> ${build_dir}/packages/rocm-bindings-compiler/src/rocm/bindings/clang/cindex.py

  # 2) copy LLVM LICENSE.TXT next to the clang bindings.
  echo "copy LLVM LICENSE.TXT into rocm.bindings.clang package"
  rm -f LICENSE.TXT
  cp ${rocm_llvm_project_dir}/LICENSE.TXT .

cd ${build_dir}

### re-raise the generator status

# All best-effort steps have run; surface the generator's exit status so a
# partial generation is still signalled to the caller. Committing the (possibly
# partial) tree is a separate concern handled by ci/internal/commit-bindings.sh.
exit "${gen_rc:-0}"
