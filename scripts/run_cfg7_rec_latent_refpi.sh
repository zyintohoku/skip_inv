#!/bin/bash

# Submit the second-stage CFG=7 reconstructed-latent FPI experiment.
#
# Experiment:
#   load rec_latents_cfg7.pt from the previous CFG-mismatch experiment,
#   invert each latent at CFG=7, then reconstruct again at CFG=7.
#
# Usage:
#   bash scripts/run_cfg7_rec_latent_refpi.sh
#   bash scripts/run_cfg7_rec_latent_refpi.sh yagi35
#
# Environment overrides:
#   SOURCE_OUTPUT=outputs/cfg_mismatch_worst_rank1_sample0270
#   SEEDS=1-10
#   OUTPUT_DIR=outputs/cfg7_rec_latent_refpi_sample0270_inv7_rec7
#   INVERSION_GUIDANCE_SCALE=7
#   RECONSTRUCTION_GUIDANCE_SCALE=7
#   METHOD=fpi
#   NUM_DDIM_STEPS=50
#   TRACE_DTYPE=float16
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
SOURCE_OUTPUT=${SOURCE_OUTPUT:-outputs/cfg_mismatch_worst_rank1_sample0270}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-}
SOURCE_REC_LATENTS_PATH=${SOURCE_REC_LATENTS_PATH:-}
SOURCE_RECONSTRUCTION_GUIDANCE_SCALE=${SOURCE_RECONSTRUCTION_GUIDANCE_SCALE:-7}
SEEDS=${SEEDS:-}
PROMPT=${PROMPT:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
INVERSION_GUIDANCE_SCALE=${INVERSION_GUIDANCE_SCALE:-7}
RECONSTRUCTION_GUIDANCE_SCALE=${RECONSTRUCTION_GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
TRACE_DTYPE=${TRACE_DTYPE:-float16}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

job_name="cfg7_rec_latent_refpi"

cmd=(python run_cfg7_rec_latent_refpi.py
    --source_output "$SOURCE_OUTPUT"
    --source_reconstruction_guidance_scale "$SOURCE_RECONSTRUCTION_GUIDANCE_SCALE"
    --inversion_guidance_scale "$INVERSION_GUIDANCE_SCALE"
    --reconstruction_guidance_scale "$RECONSTRUCTION_GUIDANCE_SCALE"
    --method "$METHOD"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --trace_dtype "$TRACE_DTYPE"
    --model_name "$MODEL_NAME")

if [ -n "$SOURCE_MANIFEST" ]; then
    cmd+=(--source_manifest "$SOURCE_MANIFEST")
fi
if [ -n "$SOURCE_REC_LATENTS_PATH" ]; then
    cmd+=(--source_rec_latents_path "$SOURCE_REC_LATENTS_PATH")
fi
if [ -n "$SEEDS" ]; then
    cmd+=(--seeds "$SEEDS")
fi
if [ -n "$PROMPT" ]; then
    cmd+=(--prompt "$PROMPT")
fi
if [ -n "$OUTPUT_DIR" ]; then
    cmd+=(--output "$OUTPUT_DIR")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "Submitting $job_name -> $selected_node (partition=$partition)"
echo "Command: ${cmd[*]}"

sbatch --job-name="$job_name" \
       --chdir="$PROJECT_DIR" \
       --nodelist="$selected_node" \
       --partition="$partition" \
       --gres=gpu:1 \
       --mem="$MEM" \
       --cpus-per-task="$CPUS_PER_TASK" \
       --output="$LOG_DIR/${job_name}.out" \
       --error="$LOG_DIR/${job_name}.err" \
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"

echo "Submitted. Check status with: squeue -u \$(whoami)"
