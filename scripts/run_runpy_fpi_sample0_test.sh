#!/usr/bin/env bash
set -euo pipefail

# Submit a run.py smoke test for prompt 0 with FPI gen/inv/rec.
#
# Usage:
#   bash scripts/run_runpy_fpi_sample0_test.sh
#   bash scripts/run_runpy_fpi_sample0_test.sh yagi35
#
# Useful overrides:
#   SEED=1 SAMPLE_IDS=0 METHOD=fpi bash scripts/run_runpy_fpi_sample0_test.sh yagi35
#   OUTPUT_DIR=outputs/runpy_fpi_sample0_seed0_test bash scripts/run_runpy_fpi_sample0_test.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
SEED=${SEED:-0}
SAMPLE_IDS=${SAMPLE_IDS:-0}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/runpy_${METHOD}_sample${SAMPLE_IDS}_seed${SEED}_test}
JOB_NAME=${JOB_NAME:-runpy_${METHOD}_sample${SAMPLE_IDS}_s${SEED}_test}

slurm_require_command sinfo
slurm_require_command sbatch
slurm_prepare_log_dir
slurm_load_node_slots NODE_SLOTS_ARRAY "$@"

selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
partition=$(slurm_partition "$selected_node")

cmd=(
    python run.py
    --method "$METHOD"
    --guidance_scale "$GUIDANCE_SCALE"
    --seed "$SEED"
    --output "$OUTPUT_DIR"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --mapping_file "$MAPPING_FILE"
    --sample_ids "$SAMPLE_IDS"
    --model_name "$MODEL_NAME"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "run.py FPI smoke test"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Method: $METHOD"
echo "Seed: $SEED"
echo "Sample ids: $SAMPLE_IDS"
echo "Output dir: $OUTPUT_DIR"
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
       --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python -c \"import torch; print(\\\"torch_cuda_available=\\\" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)\" && $quoted_cmd'"

echo "Submitted. Check status with: squeue -u $(whoami)"
echo "Logs: $LOG_DIR/${JOB_NAME}.out"
