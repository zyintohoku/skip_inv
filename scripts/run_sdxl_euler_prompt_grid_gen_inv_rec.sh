#!/bin/bash

# Submit SDXL-base Euler gen-inv-rec jobs for the custom 14 subject x 10 context grid.
#
# Usage:
#   bash scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh
#   bash scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh yagi35 yagi38
#
# Useful overrides:
#   SEEDS_SPEC=1-4 METHOD=euler bash scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh yagi35
#   STEPS=25 MAX_ITERATIONS=50 NODE_TASKS=yagi35:2,yagi38:1 bash scripts/run_sdxl_euler_prompt_grid_gen_inv_rec.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

HF_HOME=${HF_HOME:-$PROJECT_DIR/.hf_cache}

TARGET_NODES=${TARGET_NODES:-"yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"}
SEEDS_SPEC=${SEEDS_SPEC:-1-10}
PROMPT_CSV=${PROMPT_CSV:-tmp_sdxl_base_test/sdxl_custom_subject_context_prompt_grid.csv}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/sdxl_custom_subject_context_euler_fpi_gs7_seed}
RESULTS_DIR=${RESULTS_DIR:-results/sdxl_euler_prompt_grid}
RESULT_NAME=${RESULT_NAME:-sdxl_custom_subject_context_euler_fpi}
CONDA_ENV=${CONDA_ENV:-afpi}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-sdxl_euler_s}
AGG_JOB_NAME=${AGG_JOB_NAME:-sdxl_euler_agg}
JOBS_PER_NODE=${JOBS_PER_NODE:-${TASKS_PER_NODE:-1}}
TASKS_PER_NODE=$JOBS_PER_NODE
RUN_AGGREGATE=${RUN_AGGREGATE:-1}

METHOD=${METHOD:-fpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
STEPS=${STEPS:-25}
HEIGHT=${HEIGHT:-1024}
WIDTH=${WIDTH:-1024}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
MAX_ITERATIONS=${MAX_ITERATIONS:-50}
VARIANT=${VARIANT:-fp16}
NEGATIVE_PROMPT=${NEGATIVE_PROMPT:-}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
MEM=${MEM:-24G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

slurm_prepare_log_dir
mkdir -p "$HF_HOME"
AGG_OUTPUTS_DIR=$(dirname "$OUTPUT_PREFIX")
AGG_RUN_PREFIX=$(basename "$OUTPUT_PREFIX")

validate_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: $name must be a non-negative integer, got '$value'." >&2
        exit 1
    fi
}

parse_seeds() {
    local spec="$1"
    local token start end value
    local seeds=()

    IFS=',' read -ra seed_tokens <<< "$spec"
    for token in "${seed_tokens[@]}"; do
        token=${token//[[:space:]]/}
        if [ -z "$token" ]; then
            continue
        fi
        if [[ "$token" == *-* ]]; then
            start=${token%%-*}
            end=${token#*-}
            validate_nonnegative_int "seed range start" "$start"
            validate_nonnegative_int "seed range end" "$end"
            if [ "$end" -lt "$start" ]; then
                echo "Error: invalid seed range '$token'." >&2
                exit 1
            fi
            for ((value=start; value<=end; value++)); do
                seeds+=("$value")
            done
        else
            validate_nonnegative_int "seed" "$token"
            seeds+=("$token")
        fi
    done

    if [ ${#seeds[@]} -eq 0 ]; then
        echo "Error: no seeds parsed from SEEDS_SPEC='$spec'." >&2
        exit 1
    fi

    echo "${seeds[@]}"
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$PROJECT_DIR/$path"
    fi
}

slurm_require_command sinfo
slurm_require_command sbatch

validate_nonnegative_int "JOBS_PER_NODE" "$JOBS_PER_NODE"
read -ra SEEDS <<< "$(parse_seeds "$SEEDS_SPEC")"

slurm_load_node_slots NODE_SLOTS "$@"

LOCAL_FILES_ONLY_FLAG=""
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    LOCAL_FILES_ONLY_FLAG="--local_files_only"
fi

echo "Total jobs: ${#SEEDS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Project dir: $PROJECT_DIR"
echo "HF_HOME: $HF_HOME"
echo "Prompt CSV: $PROMPT_CSV"
echo "Prompt CSV absolute: $(abs_path "$PROMPT_CSV")"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Output prefix absolute: $(abs_path "$OUTPUT_PREFIX")"
echo "Results dir: $RESULTS_DIR"
echo "Method: $METHOD"
echo "Steps: $STEPS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Local files only: $LOCAL_FILES_ONLY"
echo "Jobs per node default: $JOBS_PER_NODE"
echo "Node slots: ${NODE_SLOTS[*]}"
echo ""

JOB_IDS=()
job_idx=0
for seed in "${SEEDS[@]}"; do
    selected_node=${NODE_SLOTS[$((job_idx % ${#NODE_SLOTS[@]}))]}
    partition=$(slurm_partition "$selected_node")

    name="${JOB_NAME_PREFIX}${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}"
    echo "Submitting seed=$seed -> $selected_node (partition=$partition, output=$output_dir)"

    job_id=$(sbatch --parsable \
        --job-name="$name" \
        --chdir="$PROJECT_DIR" \
        --nodelist="$selected_node" \
        --partition="$partition" \
        --gres=gpu:1 \
        --mem="$MEM" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --output="$LOG_DIR/${name}.out" \
        --error="$LOG_DIR/${name}.err" \
        --wrap="bash -c 'echo hostname=\$(hostname); echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}; export HF_HOME=\"$HF_HOME\"; export HUGGINGFACE_HUB_CACHE=\"$HF_HOME/hub\"; echo HF_HOME=\$HF_HOME; source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python -c \"import torch; print(\\\"torch_cuda_available=\\\" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)\" && python run_sdxl_euler_prompt_grid_gen_inv_rec.py --prompt_csv \"$PROMPT_CSV\" --seed $seed --output \"$output_dir\" --method $METHOD --guidance_scale $GUIDANCE_SCALE --steps $STEPS --height $HEIGHT --width $WIDTH --delta_threshold $DELTA_THRESHOLD --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD --max_iterations $MAX_ITERATIONS --variant $VARIANT --negative_prompt \"$NEGATIVE_PROMPT\" --no_progress $LOCAL_FILES_ONLY_FLAG'")
    JOB_IDS+=("$job_id")
    job_idx=$((job_idx + 1))
done

if [ "$RUN_AGGREGATE" = "1" ]; then
    dependency=$(IFS=:; echo "${JOB_IDS[*]}")
    agg_job_id=$(sbatch --parsable \
        --job-name="$AGG_JOB_NAME" \
        --chdir="$PROJECT_DIR" \
        --dependency="afterok:$dependency" \
        --mem=8G \
        --cpus-per-task=2 \
        --output="$LOG_DIR/${AGG_JOB_NAME}.out" \
        --error="$LOG_DIR/${AGG_JOB_NAME}.err" \
        --wrap="bash -c 'export HF_HOME=\"$HF_HOME\"; export HUGGINGFACE_HUB_CACHE=\"$HF_HOME/hub\"; source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python analysis/compute_sdxl_euler_prompt_grid_psnr.py --outputs_dir \"$AGG_OUTPUTS_DIR\" --run_prefix \"$AGG_RUN_PREFIX\" --results_dir \"$RESULTS_DIR\" --name \"$RESULT_NAME\" --seeds \"$SEEDS_SPEC\"'")
    echo "Submitted aggregate job: $agg_job_id"
fi

echo "Submitted seed jobs: ${JOB_IDS[*]}"
echo "Check status with: squeue -u $(whoami)"
