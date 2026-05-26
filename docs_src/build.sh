#!/usr/bin/env bash
# Minimal sphinx-build wrapper for devs who've already rendered
# index.rst (typically via a prior cmake configure). For the full
# pipeline (substitution + sphinx) see ci/docs/build.sh.
#
# Parallelism / extra flags: SPHINXOPTS env var (default "-j 8"),
# same convention used by docs_src/Makefile.
set -euo pipefail
SPHINXOPTS="${SPHINXOPTS:--j 8}"
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en \
  ${SPHINXOPTS} . _build/html
