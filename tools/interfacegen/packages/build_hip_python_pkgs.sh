#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--pre-clean)
      PRE_CLEAN=1
      shift
      ;;
    -a|--post-clean)
      POST_CLEAN=1
      shift
      ;;
    -n|--no-venv)
      NO_VENV=1
      shift
      ;;
    --no-hip)
      NO_HIP=1
      shift
      ;;
    --no-cuda)
      NO_CUDA=1
      shift
      ;;
    --no-docs)
      NO_DOCS=1
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

#echo "PRE_CLEAN=$PRE_CLEAN"
#echo "POST_CLEAN=$POST_CLEAN"
#echo "NO_VENV=$NO_VENV"
#echo "NO_HIP=$NO_HIP"
#echo "NO_CUDA=$NO_CUDA"
#echo "NO_DOCS=$NO_DOCS"

SAVED_ROCM_PATH=${ROCM_PATH}
SAVED_HIP_PLATFORM=${HIP_PLATFORM}

export ROCM_PATH=${ROCM_PATH:-"/opt/rocm"} # adjust accordingly
export HIP_PLATFORM=${HIP_PLATFORM:-"amd"}

# note: [ -z {var+x} ] evaluates to true if `var` is unset!

[ -z ${PRE_CLEAN+x} ] || rm -rf venv

alias PYTHON="python3"
if [ -z ${NO_ENV+x} ]; then
  [ ! -d "venv" ] && python3 -m venv _venv
  alias PYTHON="_venv/bin/python3"
fi
shopt -s expand_aliases

if [ -z ${NO_HIP+x} ]; then
  # build hip-python
  echo "building package hip-python"
  PKG="hip-python"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON -m build ${PKG} -n
fi
  
if [ -z ${NO_CUDA+x} ]; then
  # build hip-python-as-cuda
  echo "building package hip-python-as-cuda"
  PKG="hip-python-as-cuda"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON -m build ${PKG} -n
fi

if [ -z ${NO_DOCS+x} ]; then
  echo "building docs for package hip-python"
  # build docs
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl \
                                                    hip-python-as-cuda/dist/hip*whl
  PYTHON -m pip install -r hip-python/docs/requirements.txt
  DOCS_DIR="hip-python/docs"
  PYTHON -m sphinx -T -E -b html -d _build/doctrees -D language=en ${DOCS_DIR} ${DOCS_DIR}/_build/html
fi

export ROCM_PATH="${SAVED_ROCM_PATH}"
export HIP_PLATFORM="${SAVED_HIP_PLATFORM}"

[ -z ${POST_CLEAN+x} ] || rm -rf venv
