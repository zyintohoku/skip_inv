#!/bin/bash

# Submit the worst-prompt CFG mismatch FPI experiment.
#
# Experiment:
#   generate at CFG=7, invert at CFG=1, reconstruct at CFG=1 and CFG=7.
#
# Usage:
#   bash scripts/run_worst_prompt_cfg_mismatch_fpi.sh
#   bash scripts/run_worst_prompt_cfg_mismatch_fpi.sh yagi35
#
# Environment overrides:
#   WORST_RANK=1
#   SAMPLE_ID=270
#   SEEDS=1-10
#   OUTPUT_DIR=outputs/cfg_mismatch_worst_rank1_sample0270
#   GENERATION_GUIDANCE_SCALE=7
#   INVERSION_GUIDANCE_SCALE=1
#   RECONSTRUCTION_GUIDANCE_SCALES=1,7
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
WORST_RANK=${WORST_RANK:-1}
TOP_K=${TOP_K:-10}
SAMPLE_ID=${SAMPLE_ID:-}
PROMPT=${PROMPT:-}
SEEDS=${SEEDS:-1-10}
OUTPUT_DIR=${OUTPUT_DIR:-}
GENERATION_GUIDANCE_SCALE=${GENERATION_GUIDANCE_SCALE:-7}
INVERSION_GUIDANCE_SCALE=${INVERSION_GUIDANCE_SCALE:-1}
RECONSTRUCTION_GUIDANCE_SCALES=${RECONSTRUCTION_GUIDANCE_SCALES:-1,7}
METHOD=${METHOD:-fpi}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
TRACE_DTYPE=${TRACE_DTYPE:-float16}
WORST_CSV=${WORST_CSV:-results/fpi_gs7_seed_psnr/prompt_psnr_worst30.csv}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

job_name="cfg_mismatch_worst_r${WORST_RANK}"
if [ -n "$SAMPLE_ID" ]; then
    job_name="cfg_mismatch_s${SAMPLE_ID}"
fi

cmd=(python run_worst_prompt_cfg_mismatch_fpi.py
    --worst_rank "$WORST_RANK"
    --top_k "$TOP_K"
    --seeds "$SEEDS"
    --generation_guidance_scale "$GENERATION_GUIDANCE_SCALE"
    --inversion_guidance_scale "$INVERSION_GUIDANCE_SCALE"
    --reconstruction_guidance_scales "$RECONSTRUCTION_GUIDANCE_SCALES"
    --method "$METHOD"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --trace_dtype "$TRACE_DTYPE"
    --worst_csv "$WORST_CSV"
    --mapping_file "$MAPPING_FILE"
    --model_name "$MODEL_NAME")

if [ -n "$SAMPLE_ID" ]; then
    cmd+=(--sample_id "$SAMPLE_ID")
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
