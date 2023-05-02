#!/usr/bin/env bash

export ROCM_PATH=/opt/rocm # adjust accordingly

SAVED_PYTHONPATH="${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)" # put local path into PYTHONPATH,
                                         # as `build` will copy only the setup script and package files into a temporary dir
python3 -m build . "$@"

export PYTHONPATH="${SAVED_PYTHONPATH}"
