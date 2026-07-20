#!/bin/bash

# Submit PIE image -> VAE latent -> inversion -> reconstruction job.
#
# Default input is the previously exported best30 image directory:
#   results/fpi_gs7_seed_psnr/prompt_psnr_best30_pie_images
#
# Usage:
#   bash scripts/run_pie_image_inv_rec.sh
#   bash scripts/run_pie_image_inv_rec.sh yagi35
#
# Examples:
#   MAPPING_KEYS=311000000008 OUTPUT=outputs/pie_key311_image_inv_rec bash scripts/run_pie_image_inv_rec.sh yagi35
#   KEYS_CSV=results/fpi_gs7_seed_psnr/prompt_psnr_best30.csv bash scripts/run_pie_image_inv_rec.sh yagi35
#   IMAGE_DIR=results/fpi_gs7_seed_psnr/prompt_psnr_best30_pie_images bash scripts/run_pie_image_inv_rec.sh yagi35
#
# Environment overrides:
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:2,yagi38:1
#   IMAGE_DIR=results/fpi_gs7_seed_psnr/prompt_psnr_best30_pie_images
#   MAPPING_KEYS=311000000008,000000000074
#   SAMPLE_IDS=308,74
#   KEYS_CSV=results/fpi_gs7_seed_psnr/prompt_psnr_best30.csv
#   OUTPUT=outputs/pie_best30_image_inv_rec
#   PROMPT_FIELD=original_prompt
#   METHOD=fpi
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   CONDA_ENV=afpi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
DEFAULT_IMAGE_DIR=results/fpi_gs7_seed_psnr/prompt_psnr_best30_pie_images
IMAGE_DIR=${IMAGE_DIR:-}
MAPPING_KEYS=${MAPPING_KEYS:-}
SAMPLE_IDS=${SAMPLE_IDS:-}
KEYS_CSV=${KEYS_CSV:-}
OUTPUT=${OUTPUT:-outputs/pie_best30_image_inv_rec}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
PROMPT_FIELD=${PROMPT_FIELD:-original_prompt}
METHOD=${METHOD:-fpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-512}
LATENT_MODE=${LATENT_MODE:-mean}
SEED=${SEED:-0}
LIMIT=${LIMIT:-}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
JOB_NAME=${JOB_NAME:-pie_image_inv_rec}

if [ -z "$IMAGE_DIR" ] && [ -z "$MAPPING_KEYS" ] && [ -z "$SAMPLE_IDS" ] && [ -z "$KEYS_CSV" ]; then
    IMAGE_DIR=$DEFAULT_IMAGE_DIR
fi

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

cmd=(python run_pie_image_inv_rec.py
    --repo_root .
    --mapping_file "$MAPPING_FILE"
    --output "$OUTPUT"
    --model_name "$MODEL_NAME"
    --prompt_field "$PROMPT_FIELD"
    --method "$METHOD"
    --guidance_scale "$GUIDANCE_SCALE"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --height "$HEIGHT"
    --width "$WIDTH"
    --latent_mode "$LATENT_MODE"
    --seed "$SEED")

if [ -n "$IMAGE_DIR" ]; then
    cmd+=(--image_dir "$IMAGE_DIR")
fi
if [ -n "$MAPPING_KEYS" ]; then
    cmd+=(--mapping_keys "$MAPPING_KEYS")
fi
if [ -n "$SAMPLE_IDS" ]; then
    cmd+=(--sample_ids "$SAMPLE_IDS")
fi
if [ -n "$KEYS_CSV" ]; then
    cmd+=(--keys_csv "$KEYS_CSV")
fi
if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "PIE Image Inversion/Reconstruction Submission"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Image dir: ${IMAGE_DIR:-none}"
echo "Mapping keys: ${MAPPING_KEYS:-none}"
echo "Sample ids: ${SAMPLE_IDS:-none}"
echo "Keys CSV: ${KEYS_CSV:-none}"
echo "Output: $OUTPUT"
echo "Prompt field: $PROMPT_FIELD"
echo "Method: $METHOD, guidance_scale=$GUIDANCE_SCALE"
echo ""

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
