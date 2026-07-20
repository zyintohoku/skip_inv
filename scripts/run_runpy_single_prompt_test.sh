#!/bin/bash
set -euo pipefail

# Submit one run.py sample/prompt test to a Slurm GPU node.
#
# Usage:
#   bash /home/yan/po1/yan/skip_inv/job_scripts/scripts/run_runpy_single_prompt_test.sh yagi35
#
# Useful overrides:
#   SAMPLE_IDS=0 SEED=0 METHOD=fpi bash /path/to/skip_inv/job_scripts/scripts/run_runpy_single_prompt_test.sh yagi35
#   OUTPUT=../artifacts/outputs/run_py_sample0_seed0_fpi bash /path/to/skip_inv/job_scripts/scripts/run_runpy_single_prompt_test.sh yagi35
#   SOURCE_INIT_PREFIX=../artifacts/outputs/aidi_gs7_seed bash /path/to/skip_inv/job_scripts/scripts/run_runpy_single_prompt_test.sh yagi35

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../src" && pwd)}
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
JOB_NAME=${JOB_NAME:-runpy_s0_seed0_fpi}

METHOD=${METHOD:-fpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
SEED=${SEED:-0}
SAMPLE_IDS=${SAMPLE_IDS:-0}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
SOURCE_INIT_PREFIX=${SOURCE_INIT_PREFIX:-../artifacts/outputs/aidi_gs7_seed}
OUTPUT=${OUTPUT:-../artifacts/outputs/run_py_sample${SAMPLE_IDS}_seed${SEED}_${METHOD}}

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
    --output "$OUTPUT"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --mapping_file "$MAPPING_FILE"
    --sample_ids "$SAMPLE_IDS"
    --model_name "$MODEL_NAME"
    --source_init_prefix "$SOURCE_INIT_PREFIX"
)
printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "run.py Single Prompt Test Submission"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Job name: $JOB_NAME"
echo "Conda env: $CONDA_ENV"
echo "Sample ids: $SAMPLE_IDS"
echo "Seed: $SEED"
echo "Method: $METHOD"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Source init prefix: $SOURCE_INIT_PREFIX"
echo "Output: $OUTPUT"
echo ""

job_id=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --chdir="$PROJECT_DIR" \
    --nodelist="$selected_node" \
    --partition="$partition" \
    --gres=gpu:1 \
    --mem="$MEM" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --output="$LOG_DIR/${JOB_NAME}.out" \
    --error="$LOG_DIR/${JOB_NAME}.err" \
    --wrap="bash -c 'echo hostname=\$(hostname); echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}; source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python -c \"import torch; print(\\\"torch_cuda_available=\\\" + str(torch.cuda.is_available())); print(\\\"torch_cuda_device_count=\\\" + str(torch.cuda.device_count())); raise SystemExit(0 if torch.cuda.is_available() else 1)\" && $quoted_cmd'")

echo "Submitted job: $job_id"
echo "Check status with: squeue -u $(whoami)"
echo "Logs:"
echo "  $PROJECT_DIR/$LOG_DIR/${JOB_NAME}.out"
echo "  $PROJECT_DIR/$LOG_DIR/${JOB_NAME}.err"
