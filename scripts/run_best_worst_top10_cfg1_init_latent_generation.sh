#!/usr/bin/env bash
set -euo pipefail

# Select FPI-GS7 best/worst top10 prompts, save their 10 saved init latents,
# then generate images from those init latents with CFG=1.
#
# Usage:
#   bash scripts/run_best_worst_top10_cfg1_init_latent_generation.sh yagi35
#
# Environment overrides:
#   TOP_K=10
#   SAVED_SEEDS=1-10
#   GUIDANCE_SCALE=1
#   OUTPUT_DIR=results/fpi_gs7_seed_psnr/best_worst_top10_cfg1_init_latent_generation
#   RANKING_CSV=results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_by_sample.csv
#   SOURCE_PREFIX=outputs/aidi_gs7_seed
#   MODEL_NAME=CompVis/stable-diffusion-v1-4
#   CONDA_ENV=afpi
#   PREPARE_ONLY=0
#   SKIP_EXISTING=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TOP_K=${TOP_K:-10}
SAVED_SEEDS=${SAVED_SEEDS:-1-10}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-results/fpi_gs7_seed_psnr/best_worst_top10_cfg1_init_latent_generation}
RANKING_CSV=${RANKING_CSV:-results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_by_sample.csv}
SOURCE_PREFIX=${SOURCE_PREFIX:-outputs/aidi_gs7_seed}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
TORCH_DTYPE=${TORCH_DTYPE:-float32}
CONDA_ENV=${CONDA_ENV:-afpi}
PREPARE_ONLY=${PREPARE_ONLY:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
JOB_NAME=${JOB_NAME:-bw_top10_cfg1_init_gen}

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

cmd=(python experiments/best_worst_top10_cfg1_init_latent_generation.py
    --ranking_csv "$RANKING_CSV"
    --output_dir "$OUTPUT_DIR"
    --source_prefix "$SOURCE_PREFIX"
    --model_name "$MODEL_NAME"
    --top_k "$TOP_K"
    --seeds "$SAVED_SEEDS"
    --guidance_scale "$GUIDANCE_SCALE"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --torch_dtype "$TORCH_DTYPE"
    --device cuda
    --disable_progress_bar)

if [ "$PREPARE_ONLY" = "1" ]; then
    cmd+=(--prepare_only)
fi

if [ "$SKIP_EXISTING" = "1" ]; then
    cmd+=(--skip_existing)
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "Best/Worst Top-K CFG=1 Init-Latent Generation"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Ranking CSV: $RANKING_CSV"
echo "Top K: $TOP_K"
echo "Saved seeds: $SAVED_SEEDS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Output: $OUTPUT_DIR"
echo ""

sbatch --job-name="$JOB_NAME" \
       --chdir="$PROJECT_DIR" \
       --nodelist="$selected_node" \
       --partition="$partition" \
       --gres=gpu:1 \
       --mem="$MEM" \
       --cpus-per-task="$CPUS_PER_TASK" \
       --time="$TIME_LIMIT" \
       --output="$LOG_DIR/${JOB_NAME}.out" \
       --error="$LOG_DIR/${JOB_NAME}.err" \
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"

echo "Submitted. Check status with: squeue -u \$(whoami)"
echo "Logs: $LOG_DIR/${JOB_NAME}.out"
