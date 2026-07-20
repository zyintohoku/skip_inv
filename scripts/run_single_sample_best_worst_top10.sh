#!/bin/bash

# Submit one sample_id per GPU job for best/worst top10 evaluation.
# Usage:
#   bash scripts/run_single_sample_best_worst_top10.sh
#   bash scripts/run_single_sample_best_worst_top10.sh yagi36
#   bash scripts/run_single_sample_best_worst_top10.sh yagi37 yagi40 yagi45

PROJECT_DIR=${PROJECT_DIR:-$PWD}

SAMPLE_IDS=(
  134 363 589 189 78 400 482 572 409 515
  575 152 544 474 347 95 236 576 332 491
)

NUM_TRIALS=50
SEED_START=0
SEED_STRIDE=1
GUIDANCE_SCALE=7
DDIM_STEPS=50
DELTA_THRESHOLD=5e-12
OUTPUT_BASE=outputs/reconstruction/sample_sweeps_top10

if [ $# -eq 1 ]; then
    SINGLE_NODE=$1
    echo "Using single node for all jobs: $SINGLE_NODE"
    for i in "${!SAMPLE_IDS[@]}"; do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
elif [ $# -ge 1 ]; then
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
else
    echo "Searching for available nodes..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(29|33|35|36|37|40|45)" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)
    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ] || [ -z "${NODE_ARRAY[0]}" ]; then
        echo "❌ No available nodes found!"
        exit 1
    fi
    echo "✅ Found ${#NODE_ARRAY[@]} available nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
fi
echo ""

for i in "${!SAMPLE_IDS[@]}"; do
    sample_id="${SAMPLE_IDS[$i]}"
    node_idx=$((i % ${#NODE_ARRAY[@]}))
    selected_node="${NODE_ARRAY[$node_idx]}"
    job_name="ss_${sample_id}"
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

    echo "  ├─ Submitting $job_name → $selected_node (sample_id=$sample_id)"

    sbatch --job-name=$job_name \
           --nodelist=$selected_node \
           --partition=$partition \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --output=log/${job_name}.out \
           --error=log/${job_name}.err \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python run_single_sample.py --sample_id ${sample_id} --method aidi --guidance_scale ${GUIDANCE_SCALE} --num_of_ddim_steps ${DDIM_STEPS} --delta_threshold ${DELTA_THRESHOLD} --num_trials ${NUM_TRIALS} --seed_start ${SEED_START} --seed_stride ${SEED_STRIDE} --output ${OUTPUT_BASE}'"
done

echo ""
echo "✨ All ${#SAMPLE_IDS[@]} jobs submitted!"
echo "Each sample uses identical init-latent seed schedule:"
echo "  seed = SEED_START + trial_idx * SEED_STRIDE (num_trials=${NUM_TRIALS})"
echo "Check status: squeue -u \$(whoami)"
