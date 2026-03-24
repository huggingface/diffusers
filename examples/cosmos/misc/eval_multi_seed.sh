#!/usr/bin/env bash
# Each process is pinned to one GPU and writes outputs to output_dir/seed_<N>/.

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

lora_dir='outputs/gr1-r8/checkpoint-399' # The directory that contains lora weights after training
data_dir='dream_gen_benchmark/gr1_object'
revision='post-trained'

if [ "$lora_dir" != "None" ]; then
  output_dir=$lora_dir/results-fuse
else
  output_dir=outputs/$revision/results
fi
SEEDS=(1 2 3 4)          # one per GPU, run in parallel
gpu_offset=0

PIDS=()

cleanup() {
    echo "Killing all child processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    OUT="${output_dir}/seed_${SEED}"
    mkdir -p "$OUT"

    python_args=(
      --seed $SEED
      --data_dir $data_dir
      --revision diffusers/base/$revision
      --height 432 --width 768
      --output_dir $OUT
    )

    if [ "$lora_dir" != "None" ]; then
      python_args+=(--lora_dir $lora_dir)
    fi


    GPU_ID=$(( i + gpu_offset ))
    CUDA_VISIBLE_DEVICES=$GPU_ID python ./scripts/eval_cosmos_predict25_lora.py "${python_args[@]}" > "${OUT}/run.log" 2>&1 &

    PIDS+=($!)
    echo "Launched seed=$SEED on GPU $GPU_ID (PID ${PIDS[-1]}), logs: ${OUT}/run.log"
done

echo "Waiting for all ${#PIDS[@]} jobs..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: seed=${SEEDS[$i]} (GPU $i, PID ${PIDS[$i]}) failed — check ${output_dir}/seed_${SEEDS[$i]}/run.log"
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -eq 0 ]]; then
    echo "All jobs completed successfully."
else
    echo "$FAILED job(s) failed."
    exit 1
fi
