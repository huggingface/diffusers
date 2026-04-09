#!/bin/bash
# Run profiling across all pipelines in eager and compile (regional) modes.
#
# Usage:
#   bash profiling/run_profiling.sh
#   bash profiling/run_profiling.sh --output_dir my_results

set -euo pipefail

OUTPUT_DIR="profiling_results"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
NUM_STEPS=2
# PIPELINES=("flux" "flux2" "wan" "ltx2" "qwenimage")
PIPELINES=("wan")
MODES=("eager" "compile")

for pipeline in "${PIPELINES[@]}"; do
    for mode in "${MODES[@]}"; do
        echo "============================================================"
        echo "Profiling: ${pipeline} | mode: ${mode}"
        echo "============================================================"

        COMPILE_ARGS=""
        if [ "$mode" = "compile" ]; then
            COMPILE_ARGS="--compile_regional --compile_fullgraph --compile_mode default"
        fi

        python profiling/profiling_pipelines.py \
            --pipeline "$pipeline" \
            --mode "$mode" \
            --output_dir "$OUTPUT_DIR" \
            --num_steps "$NUM_STEPS" \
            $COMPILE_ARGS

        echo ""
    done
done

echo "============================================================"
echo "All traces saved to: ${OUTPUT_DIR}/"
echo "============================================================"
