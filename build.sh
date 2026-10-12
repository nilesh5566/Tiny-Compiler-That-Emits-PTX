#!/usr/bin/env bash
# build.sh - configure, build and test KernC on Linux / macOS / Colab.
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"
./build/bin/kernc_unit_tests
python3 tools/run_tests.py "$@"
