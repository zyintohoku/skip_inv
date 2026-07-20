#!/usr/bin/env bash
set -euo pipefail

# Generate images from the 20 selected FPI inversion latents using the best-PSNR top1 prompt.
#
# Usage:
#   bash scripts/run_selected_inv_latent_best_prompt_generation.sh
#   bash scripts/run_selected_inv_latent_best_prompt_generation.sh yagi35
#
# Optional env overrides:
#   PROMPT="..." OUTPUT_DIR=... bash scripts/run_selected_inv_latent_best_prompt_generation.sh yagi35

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV="${CONDA_ENV:-afpi}"
MANIFEST="${MANIFEST:-results/all_prompt_seed_clip_scores/selected_prompt_seed_examples/selected_prompt_seed_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/all_prompt_seed_clip_scores/selected_prompt_seed_examples/best_prompt_top1_from_inv_latents}"
MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
NUM_DDIM_STEPS="${NUM_DDIM_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7}"
PROMPT="${PROMPT:-a mountain is covered in clouds and snow}"
IMAGE_TAG="${IMAGE_TAG:-best_prompt_top1}"
LOG_DIR="${LOG_DIR:-log}"
JOB_NAME="${JOB_NAME:-selected_inv_best_prompt}"
PARTITION="${PARTITION:-}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

slurm_prepare_log_dir
slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
NODE=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")

if [[ -z "$PARTITION" ]]; then
  PARTITION="$(slurm_partition "$NODE")"
fi

if [[ -z "$PARTITION" ]]; then
  echo "Cannot determine partition, using default 48-4"
  PARTITION="48-4"
fi

cmd=(python generate_selected_inv_latent_uncond.py
  --manifest "$MANIFEST"
  --output "$OUTPUT_DIR"
  --model_name "$MODEL_NAME"
  --prompt "$PROMPT"
  --image_tag "$IMAGE_TAG"
  --num_of_ddim_steps "$NUM_DDIM_STEPS"
  --guidance_scale "$GUIDANCE_SCALE"
  --device cuda
  --make_montage)

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "Submitting $JOB_NAME${NODE:+ -> $NODE} (partition=$PARTITION)"
echo "Prompt: $PROMPT"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Output: $OUTPUT_DIR"

sbatch --job-name="$JOB_NAME" \
       --chdir="$PROJECT_DIR" \
       ${NODE:+--nodelist="$NODE"} \
       --partition="$PARTITION" \
       --gres="gpu:${GPUS}" \
       --mem="$MEM" \
       --cpus-per-task="$CPUS_PER_TASK" \
       --time="$TIME_LIMIT" \
       --output="$LOG_DIR/${JOB_NAME}.out" \
       --error="$LOG_DIR/${JOB_NAME}.err" \
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"
