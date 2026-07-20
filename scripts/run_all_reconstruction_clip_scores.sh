#!/bin/bash

# Submit CLIP image-score computation for all FPI-GS7 gen/rec image pairs.
#
# Default input:
#   results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv
# Expected size:
#   7000 rows = 700 prompts x 10 seeds
#
# Usage:
#   bash scripts/run_all_reconstruction_clip_scores.sh
#   bash scripts/run_all_reconstruction_clip_scores.sh yagi35
#
# Environment overrides:
#   OUTPUT_DIR=results/all_prompt_seed_clip_scores
#   DETAIL_CSV=results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv
#   BATCH_SIZE=128
#   BACKEND=auto
#   CLIP_MODEL=ViT-B/32
#   HF_CLIP_MODEL=openai/clip-vit-base-patch32
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
OUTPUT_DIR=${OUTPUT_DIR:-results/all_prompt_seed_clip_scores}
DETAIL_CSV=${DETAIL_CSV:-results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv}
BATCH_SIZE=${BATCH_SIZE:-64}
BACKEND=${BACKEND:-auto}
CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
HF_CLIP_MODEL=${HF_CLIP_MODEL:-openai/clip-vit-base-patch32}
EXPECTED_ROWS=${EXPECTED_ROWS:-7000}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-32G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
JOB_NAME=${JOB_NAME:-all_clip_scores}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

cmd=(python analysis/compute_all_reconstruction_clip_scores.py
    --detail_csv "$DETAIL_CSV"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --backend "$BACKEND"
    --clip_model "$CLIP_MODEL"
    --hf_clip_model "$HF_CLIP_MODEL"
    --expected_rows "$EXPECTED_ROWS"
    --device cuda)

if [ "$RESUME" = "1" ]; then
    cmd+=(--resume)
else
    cmd+=(--no_resume)
fi
if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "Submitting $JOB_NAME -> $selected_node (partition=$partition)"
echo "Output dir: $OUTPUT_DIR"

sbatch --job-name="$JOB_NAME" \
       --chdir="$PROJECT_DIR" \
       --nodelist="$selected_node" \
       --partition="$partition" \
       --gres=gpu:1 \
       --mem="$MEM" \
       --cpus-per-task="$CPUS_PER_TASK" \
       --output="$LOG_DIR/${JOB_NAME}.out" \
       --error="$LOG_DIR/${JOB_NAME}.err" \
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"

echo "Submitted. Check status with: squeue -u \$(whoami)"
