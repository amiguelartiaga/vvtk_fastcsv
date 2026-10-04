#!/usr/bin/env bash
set -euo pipefail
python benchmarks/benchmark.py --scale 0.05 --repeats 2 --threads 0
