#!/usr/bin/env bash
set -euo pipefail

# Submit the most-sensitive prompt pressure run with DDIM inversion/reconstruction.
#
# Usage:
#   bash scripts/run_sensitive_prompt_pressure_ddim_inv_rec.sh
#   bash scripts/run_sensitive_prompt_pressure_ddim_inv_rec.sh yagi38
#
# Useful overrides:
#   SEEDS=0-99 bash scripts/run_sensitive_prompt_pressure_ddim_inv_rec.sh yagi38
#   DRY_RUN=1 bash scripts/run_sensitive_prompt_pressure_ddim_inv_rec.sh yagi38

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}

SEEDS=${SEEDS:-}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-../artifacts/outputs/prompt_group_top1_fpi_100/sensitive/manifest.csv}
RESULTS_DIR=${RESULTS_DIR:-../artifacts/results/fpi_gs7_seed_psnr}
OUTPUT_DIR=${OUTPUT_DIR:-../artifacts/outputs/sensitive_prompt_pressure_ddim_inv_rec}
SAVE_IMAGES=${SAVE_IMAGES:-1}
SAVE_LATENTS=${SAVE_LATENTS:-1}
DISABLE_PROGRESS_BAR=${DISABLE_PROGRESS_BAR:-1}
DRY_RUN=${DRY_RUN:-0}

if [ "$DRY_RUN" != "1" ]; then
    slurm_require_command sinfo
    slurm_require_command sbatch
elif [ "$#" -eq 0 ] && [ -z "${NODE_TASKS:-}" ] && [ -z "${NODE_SLOTS:-}" ]; then
    slurm_require_command sinfo
fi
slurm_prepare_log_dir
slurm_load_node_slots NODE_SLOTS_ARRAY "$@"

selected_node=$(slurm_node_for_index 0 "${NODE_SLOTS_ARRAY[@]}")
if command -v sinfo >/dev/null 2>&1; then
    partition=$(slurm_partition "$selected_node")
else
    partition=$DEFAULT_PARTITION
fi
job_name="sensitive_pressure_ddim"

cmd=(
    python experiments/run_sensitive_prompt_pressure_ddim_inv_rec.py
    --source_manifest "$SOURCE_MANIFEST"
    --results_dir "$RESULTS_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_name "$MODEL_NAME"
    --guidance_scale "$GUIDANCE_SCALE"
    --num_of_ddim_steps "$NUM_DDIM_STEPS"
    --delta_threshold "$DELTA_THRESHOLD"
    --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
    --device cuda
)
if [ -n "$SEEDS" ]; then
    cmd+=(--seeds "$SEEDS")
fi
if [ "$DISABLE_PROGRESS_BAR" = "1" ]; then
    cmd+=(--disable_progress_bar)
fi
if [ "$SAVE_IMAGES" != "1" ]; then
    cmd+=(--no-save_images)
fi
if [ "$SAVE_LATENTS" != "1" ]; then
    cmd+=(--no-save_latents)
fi
printf -v quoted_cmd '%q ' "${cmd[@]}"

echo "============================================================"
echo "Sensitive prompt pressure + DDIM inversion/reconstruction"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Node: $selected_node (partition=$partition)"
echo "Source manifest: $SOURCE_MANIFEST"
echo "Output dir: $OUTPUT_DIR"
echo "Seeds: ${SEEDS:-source manifest seeds}"
echo "Dry run: $DRY_RUN"
echo "Command: $quoted_cmd"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run only; not submitting."
else
    sbatch --job-name="$job_name" \
           --chdir="$PROJECT_DIR" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --time="$TIME_LIMIT" \
           --output="$LOG_DIR/${job_name}.out" \
           --error="$LOG_DIR/${job_name}.err" \
           --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && { echo hostname=\$(hostname); echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}; echo SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-unset}; nvidia-smi || true; } && python -c \"import torch; print(\\\"torch_cuda_available=\\\" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)\" && $quoted_cmd'"
    echo "Submitted. Check status with: squeue -u $(whoami)"
    echo "Logs: $LOG_DIR/${job_name}.{out,err}"
fi
