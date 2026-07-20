#!/bin/bash

# Submit paraphrase prompt-grid FPI gen-inv-rec jobs over seeds 1..10.
# Each seed job samples one initial latent once, saves it, and reuses it for
# every prompt in that seed's prompt CSV.
#
# Usage:
#   bash scripts/run_paraphrase_prompt_grid_fpi_shared_init_gen_inv_rec.sh
#   bash scripts/run_paraphrase_prompt_grid_fpi_shared_init_gen_inv_rec.sh yagi38 yagi40
#
# Useful overrides:
#   SEEDS_SPEC=1-4 bash scripts/run_paraphrase_prompt_grid_fpi_shared_init_gen_inv_rec.sh yagi38
#   JOBS_PER_NODE=2 bash scripts/run_paraphrase_prompt_grid_fpi_shared_init_gen_inv_rec.sh yagi38
#   NODE_JOB_COUNTS=2,1 bash scripts/run_paraphrase_prompt_grid_fpi_shared_init_gen_inv_rec.sh yagi38 yagi40

PROJECT_DIR=${PROJECT_DIR:-$PWD}

TARGET_NODES=${TARGET_NODES:-"yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"}
SEEDS_SPEC=${SEEDS_SPEC:-1-10}
PROMPT_CSV=${PROMPT_CSV:-results/fpi_gs7_seed_psnr/paraphrase_prompt_grid/paraphrase_prompt_grid.csv}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/paraphrase_prompt_grid_fpi_shared_init_gs7_seed}
RESULTS_DIR=${RESULTS_DIR:-results/fpi_gs7_seed_psnr/paraphrase_prompt_grid_shared_init}
RESULT_NAME=${RESULT_NAME:-paraphrase_prompt_grid_fpi_shared_init}
CONDA_ENV=${CONDA_ENV:-afpi}
JOBS_PER_NODE=${JOBS_PER_NODE:-1}
NODE_JOB_COUNTS=${NODE_JOB_COUNTS:-}
RUN_AGGREGATE=${RUN_AGGREGATE:-1}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-para_fpi_shared_s}
AGG_JOB_NAME=${AGG_JOB_NAME:-para_fpi_shared_agg}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-float32}

AGG_OUTPUTS_DIR=$(dirname "$OUTPUT_PREFIX")
AGG_RUN_PREFIX=$(basename "$OUTPUT_PREFIX")
mkdir -p log

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

node_job_count() {
    local node_idx="$1"
    local default_count="$JOBS_PER_NODE"
    local counts

    if [ -z "$NODE_JOB_COUNTS" ]; then
        echo "$default_count"
        return
    fi

    IFS=',' read -ra counts <<< "$NODE_JOB_COUNTS"
    if [ "$node_idx" -lt "${#counts[@]}" ] && [ -n "${counts[$node_idx]}" ]; then
        echo "${counts[$node_idx]}"
        return
    fi

    echo "$default_count"
}

if ! command -v sinfo >/dev/null 2>&1; then
    echo "Error: sinfo not found. Run this script on a Slurm login node." >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch not found. Run this script on a Slurm login node." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Searching for available nodes in TARGET_NODES: $TARGET_NODES"
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | awk -v nodes="$TARGET_NODES" '
        BEGIN {
            split(nodes, arr, ",");
            for (i in arr) allow[arr[i]] = 1;
        }
        $2 ~ /idle|mix/ && ($1 in allow) { print $1 }
    ' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"
    if [ ${#NODE_ARRAY[@]} -eq 0 ]; then
        echo "No available nodes found. Current target-node status:"
        sinfo -N -h -o "%N %P %t" | awk -v nodes="$TARGET_NODES" '
            BEGIN {
                split(nodes, arr, ",");
                for (i in arr) allow[arr[i]] = 1;
            }
            ($1 in allow) { print }
        '
        exit 1
    fi
else
    NODE_ARRAY=("$@")
fi

validate_nonnegative_int "JOBS_PER_NODE" "$JOBS_PER_NODE"
read -ra SEEDS <<< "$(parse_seeds "$SEEDS_SPEC")"

NODE_SLOTS=()
for node_idx in "${!NODE_ARRAY[@]}"; do
    node=${NODE_ARRAY[$node_idx]}
    jobs_for_node=$(node_job_count "$node_idx")
    validate_nonnegative_int "job count for $node" "$jobs_for_node"
    if [ "$jobs_for_node" -eq 0 ]; then
        echo "Skipping node=$node because job count is 0"
        continue
    fi
    for ((slot=0; slot<jobs_for_node; slot++)); do
        NODE_SLOTS+=("$node")
    done
done

if [ ${#NODE_SLOTS[@]} -eq 0 ]; then
    echo "Error: no node slots selected. Check JOBS_PER_NODE or NODE_JOB_COUNTS." >&2
    exit 1
fi

echo "Total jobs: ${#SEEDS[@]}"
echo "Seeds: ${SEEDS[*]}"
echo "Project dir: $PROJECT_DIR"
echo "Prompt CSV: $PROMPT_CSV"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Results dir: $RESULTS_DIR"
echo "Node slots: ${NODE_SLOTS[*]}"
echo ""

JOB_IDS=()
job_idx=0
for seed in "${SEEDS[@]}"; do
    selected_node=${NODE_SLOTS[$((job_idx % ${#NODE_SLOTS[@]}))]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    if [ -z "$partition" ]; then
        partition="48-4"
    fi

    name="${JOB_NAME_PREFIX}${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}"
    echo "Submitting seed=$seed -> $selected_node (partition=$partition, output=$output_dir)"

    job_id=$(sbatch --parsable \
        --job-name="$name" \
        --chdir="$PROJECT_DIR" \
        --nodelist="$selected_node" \
        --partition="$partition" \
        --gres=gpu:1 \
        --mem=32G \
        --cpus-per-task=4 \
        --output="log/${name}.out" \
        --error="log/${name}.err" \
        --wrap="bash -c 'echo hostname=\$(hostname); echo CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}; source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python -c \"import torch; print(\\\"torch_cuda_available=\\\" + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)\" && python run_prompt_ablation_fpi_gen_inv_rec_shared_init.py --prompt_csv $PROMPT_CSV --method $METHOD --guidance_scale $GUIDANCE_SCALE --seed $seed --output $output_dir --num_of_ddim_steps $NUM_DDIM_STEPS --delta_threshold $DELTA_THRESHOLD --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD --device $DEVICE --torch_dtype $TORCH_DTYPE --disable_progress_bar'")
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
        --output="log/${AGG_JOB_NAME}.out" \
        --error="log/${AGG_JOB_NAME}.err" \
        --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python analysis/compute_prompt_grid_fpi_psnr.py --outputs_dir $AGG_OUTPUTS_DIR --run_prefix $AGG_RUN_PREFIX --results_dir $RESULTS_DIR --name $RESULT_NAME --seeds $SEEDS_SPEC'")
    echo "Submitted aggregate job: $agg_job_id"
fi

echo "Submitted seed jobs: ${JOB_IDS[*]}"
echo "Check status with: squeue -u $(whoami)"
