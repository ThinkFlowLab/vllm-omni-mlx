#!/bin/sh
# Run the test suite one file per process, so each file's checkpoints (and
# their Metal buffers) return to the OS when its process exits — a single
# `pytest tests/` keeps every weight-gated class's model resident for the
# whole run, which stacks multiple multi-GiB checkpoints and can swamp a
# 16 GB Mac. Skips the heavy Base+donor batteries unless --heavy is given
# (equivalent to VOMX_HEAVY_TESTS=1); they also run per-file here.
#
#   scripts/run_tests.sh              # light suite, one process per file
#   scripts/run_tests.sh --heavy      # include the heavy batteries
#   scripts/run_tests.sh tests/test_tts_generate.py   # specific files
#
# PYTHON overrides the interpreter (default .venv/bin/python); pytest is used
# when importable, else unittest. HF_HUB_OFFLINE defaults to 1 (weight-gated
# loads resolve from the local cache; set HF_HUB_OFFLINE=0 to override).

set -u

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

HEAVY=0
files=""
for arg in "$@"; do
    case "$arg" in
        --heavy) HEAVY=1 ;;
        *) files="$files $arg" ;;
    esac
done
[ -n "$files" ] || files="$(echo tests/test_*.py)"
[ "$HEAVY" = 1 ] && export VOMX_HEAVY_TESTS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if "$PYTHON" -m pytest --version >/dev/null 2>&1; then
    run_one() { "$PYTHON" -m pytest "$1" -q; }
else
    run_one() { "$PYTHON" -m unittest "$(echo "${1%.py}" | tr '/' '.')"; }
fi

failed=""
for f in $files; do
    printf '\n=== %s ===\n' "$f"
    if run_one "$f"; then :; else
        failed="$failed $f"
    fi
done

printf '\n=== summary ===\n'
if [ -n "$failed" ]; then
    printf 'FAILED:%s\n' "$failed"
    exit 1
fi
printf 'all files passed\n'
