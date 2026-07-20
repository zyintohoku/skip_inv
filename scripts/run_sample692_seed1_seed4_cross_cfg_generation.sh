#!/bin/bash

# Submit cross-CFG interpolation generation for sample 692 seed1/seed4.
#
# This only reads saved boundary inv_latents and generates images:
#   1. existing cfg7 FPI inv boundary -> generate with cfg=1
#   2. cfg1-predicted FPI inv boundary -> generate with cfg=7
#
# Usage:
#   bash scripts/run_sample692_seed1_seed4_cross_cfg_generation.sh
#   bash scripts/run_sample692_seed1_seed4_cross_cfg_generation.sh yagi35
#
# Environment overrides:
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:1
#   OUTPUT=outputs/sample692_seed1_seed4_cross_cfg_generation
#   EXISTING_INV_BOUNDARY_PATH=outputs/sample692_seed1_seed4_inv_slerp/existing_fpi_gs7_inv_boundary/boundary_latents.pt
#   CFG1_INV_BOUNDARY_PATH=outputs/sample692_seed1_seed4_inv_slerp/cfg1_fpi_from_gen_boundary/cfg1_boundary_inv_latents.pt
#   EXISTING_INV_GENERATION_GUIDANCE_SCALE=1
#   CFG1_INV_GENERATION_GUIDANCE_SCALE=7
#   ALPHAS=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
#   CONDA_ENV=afpi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

SAMPLE_ID=${SAMPLE_ID:-692}
SEED_A=${SEED_A:-1}
SEED_B=${SEED_B:-4}
OUTPUT=${OUTPUT:-outputs/sample692_seed1_seed4_cross_cfg_generation}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
EXISTING_INV_BOUNDARY_PATH=${EXISTING_INV_BOUNDARY_PATH:-outputs/sample692_seed1_seed4_inv_slerp/existing_fpi_gs7_inv_boundary/boundary_latents.pt}
CFG1_INV_BOUNDARY_PATH=${CFG1_INV_BOUNDARY_PATH:-outputs/sample692_seed1_seed4_inv_slerp/cfg1_fpi_from_gen_boundary/cfg1_boundary_inv_latents.pt}
EXISTING_INV_GENERATION_GUIDANCE_SCALE=${EXISTING_INV_GENERATION_GUIDANCE_SCALE:-1}
CFG1_INV_GENERATION_GUIDANCE_SCALE=${CFG1_INV_GENERATION_GUIDANCE_SCALE:-7}
NUM_INTERPOLATION_POINTS=${NUM_INTERPOLATION_POINTS:-11}
ALPHAS=${ALPHAS:-}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
TRACE_DTYPE=${TRACE_DTYPE:-float16}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
JOB_NAME=${JOB_NAME:-sample692_cross_cfg}

slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

cmd=(python run_sample692_seed1_seed4_cross_cfg_generation.py
    --sample_id "$SAMPLE_ID"
    --seed_a "$SEED_A"
    --seed_b "$SEED_B"
    --mapping_file "$MAPPING_FILE"
    --existing_inv_boundary_path "$EXISTING_INV_BOUNDARY_PATH"
    --cfg1_inv_boundary_path "$CFG1_INV_BOUNDARY_PATH"
    --output "$OUTPUT"
    --model_name "$MODEL_NAME"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --num_interpolation_points "$NUM_INTERPOLATION_POINTS"
    --existing_inv_generation_guidance_scale "$EXISTING_INV_GENERATION_GUIDANCE_SCALE"
    --cfg1_inv_generation_guidance_scale "$CFG1_INV_GENERATION_GUIDANCE_SCALE"
    --trace_dtype "$TRACE_DTYPE")

if [ -n "$ALPHAS" ]; then
    cmd+=(--alphas "$ALPHAS")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "Sample 692 Seed1/Seed4 Cross-CFG Generation Submission"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Output: $OUTPUT"
echo "Existing boundary: $EXISTING_INV_BOUNDARY_PATH -> cfg=$EXISTING_INV_GENERATION_GUIDANCE_SCALE"
echo "CFG1 boundary: $CFG1_INV_BOUNDARY_PATH -> cfg=$CFG1_INV_GENERATION_GUIDANCE_SCALE"
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
