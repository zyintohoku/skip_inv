#!/usr/bin/env bash
set -euo pipefail

# Submit repeated gen/inv/rec trials for best, worst, and seed-sensitive prompts.
#
# Usage:
#   bash scripts/run_prompt_group_fpi_100.sh
#   bash scripts/run_prompt_group_fpi_100.sh yagi35 yagi38 yagi39
#   NODE_TASKS=yagi35:1,yagi38:1,yagi39:1 bash scripts/run_prompt_group_fpi_100.sh
#
# Useful overrides:
#   PROMPT_GROUPS=best,worst SEEDS=0-99 TOP_K=1 bash scripts/run_prompt_group_fpi_100.sh yagi35 yagi38
#   OUTPUT_DIR=outputs/prompt_group_top1_fpi_100 DRY_RUN=1 bash scripts/run_prompt_group_fpi_100.sh yagi35

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
TIME_LIMIT=${TIME_LIMIT:-72:00:00}

PROMPT_GROUPS=${PROMPT_GROUPS:-best,worst,sensitive}
SEEDS=${SEEDS:-0-99}
TOP_K=${TOP_K:-1}
METHOD=${METHOD:-fpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
RESULTS_DIR=${RESULTS_DIR:-../artifacts/results/fpi_gs7_seed_psnr}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/prompt_group_top1_fpi_100}
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

IFS=',' read -r -a GROUP_ARRAY <<< "$PROMPT_GROUPS"
if [ "${#GROUP_ARRAY[@]}" -eq 0 ]; then
    echo "Error: PROMPT_GROUPS is empty." >&2
    exit 1
fi

echo "============================================================"
echo "Prompt group FPI repeated gen/inv/rec"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Groups: $PROMPT_GROUPS"
echo "Seeds: $SEEDS"
echo "Prompts per group: $TOP_K"
echo "Nodes in rotation: ${NODE_SLOTS_ARRAY[*]}"
echo "Dry run: $DRY_RUN"
echo ""

submitted_jobs=()
for idx in "${!GROUP_ARRAY[@]}"; do
    group=${GROUP_ARRAY[$idx]}
    group=${group//[[:space:]]/}
    if [ -z "$group" ]; then
        continue
    fi
    case "$group" in
        best|worst|sensitive) ;;
        *)
            echo "Error: invalid prompt group '$group'. Use best, worst, or sensitive." >&2
            exit 1
            ;;
    esac

    selected_node=$(slurm_node_for_index "$idx" "${NODE_SLOTS_ARRAY[@]}")
    if command -v sinfo >/dev/null 2>&1; then
        partition=$(slurm_partition "$selected_node")
    else
        partition=$DEFAULT_PARTITION
    fi
    job_name="prompt_${group}_${METHOD}_100"

    cmd=(
        python experiments/run_prompt_group_gen_inv_rec.py
        --group "$group"
        --results_dir "$RESULTS_DIR"
        --output_dir "$OUTPUT_DIR"
        --model_name "$MODEL_NAME"
        --method "$METHOD"
        --guidance_scale "$GUIDANCE_SCALE"
        --num_of_ddim_steps "$NUM_DDIM_STEPS"
        --delta_threshold "$DELTA_THRESHOLD"
        --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
        --top_k "$TOP_K"
        --seeds "$SEEDS"
        --device cuda
    )
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

    echo "Group: $group"
    echo "  Node: $selected_node (partition=$partition)"
    echo "  Job: $job_name"
    echo "  Command: $quoted_cmd"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  Dry run only; not submitting."
    else
        sbatch_output=$(
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
        )
        echo "  $sbatch_output"
        submitted_jobs+=("$sbatch_output")
    fi
    echo ""
done

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete."
else
    echo "Submitted ${#submitted_jobs[@]} jobs. Check status with: squeue -u $(whoami)"
    echo "Logs: $LOG_DIR/prompt_<group>_${METHOD}_100.{out,err}"
fi
