#!/bin/bash
set -euo pipefail

# Submit 10 shared-initial-latent runs with seeds 1..10.
# Each job calls root-level run.py; run.py samples one initial latent per seed
# and reuses it for every prompt in that job.
#
# Usage:
#   bash scripts/run_shared_init_gs7_seed_1_10.sh
#   bash scripts/run_shared_init_gs7_seed_1_10.sh yagi35 yagi38
#   bash scripts/run_shared_init_gs7_seed_1_10.sh yagi35:2 yagi38:1
#
# Useful overrides:
#   METHOD=aidi bash scripts/run_shared_init_gs7_seed_1_10.sh yagi35
#   SEEDS_SPEC=1-4 bash scripts/run_shared_init_gs7_seed_1_10.sh yagi35
#   OUTPUT_PREFIX=outputs/fpi_gs7_shared_init_seed bash scripts/run_shared_init_gs7_seed_1_10.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"

CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
SEEDS_SPEC=${SEEDS_SPEC:-1-10}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
SAMPLE_IDS=${SAMPLE_IDS:-all}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
SOURCE_INIT_PREFIX=${SOURCE_INIT_PREFIX:-outputs/aidi_gs7_seed}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/${METHOD}_gs${GUIDANCE_SCALE}_shared_init_seed}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-${METHOD}_gs${GUIDANCE_SCALE}_shared_s}

parse_int_spec() {
    local spec=$1
    local token start end value
    local values=()

    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        token=${token//[[:space:]]/}
        [ -z "$token" ] && continue

        if [[ "$token" == *-* ]]; then
            start=${token%%-*}
            end=${token#*-}
            if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || [ "$end" -lt "$start" ]; then
                echo "Error: invalid integer range '$token'." >&2
                exit 1
            fi
            for ((value = start; value <= end; value++)); do
                values+=("$value")
            done
        else
            if ! [[ "$token" =~ ^[0-9]+$ ]]; then
                echo "Error: invalid integer token '$token'." >&2
                exit 1
            fi
            values+=("$token")
        fi
    done

    if [ "${#values[@]}" -eq 0 ]; then
        echo "Error: no values parsed from '$spec'." >&2
        exit 1
    fi

    echo "${values[@]}"
}

slurm_require_command sinfo
slurm_require_command sbatch
slurm_prepare_log_dir
slurm_load_node_slots NODE_SLOTS_ARRAY "$@"
read -ra SEEDS <<< "$(parse_int_spec "$SEEDS_SPEC")"

echo "Project dir: $PROJECT_DIR"
echo "Seeds: ${SEEDS[*]}"
echo "Method: $METHOD"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Mapping file: $MAPPING_FILE"
echo "Sample ids: $SAMPLE_IDS"
echo "Source init prefix: $SOURCE_INIT_PREFIX"
echo "Node slots: ${NODE_SLOTS_ARRAY[*]}"
echo ""

JOB_IDS=()
for idx in "${!SEEDS[@]}"; do
    seed=${SEEDS[$idx]}
    node=$(slurm_node_for_index "$idx" "${NODE_SLOTS_ARRAY[@]}")
    partition=$(slurm_partition "$node")
    job_name="${JOB_NAME_PREFIX}${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}"

    cmd=(
        python run.py
        --method "$METHOD"
        --guidance_scale "$GUIDANCE_SCALE"
        --seed "$seed"
        --output "$output_dir"
        --num_of_ddim_steps "$NUM_DDIM_STEPS"
        --delta_threshold "$DELTA_THRESHOLD"
        --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
        --mapping_file "$MAPPING_FILE"
        --sample_ids "$SAMPLE_IDS"
        --model_name "$MODEL_NAME"
        --source_init_prefix "$SOURCE_INIT_PREFIX"
    )
    printf -v quoted_cmd '%q ' "${cmd[@]}"

    echo "Submitting seed=$seed -> $node (partition=$partition, output=$output_dir)"
    job_id=$(sbatch --parsable \
        --job-name="$job_name" \
        --chdir="$PROJECT_DIR" \
        --nodelist="$node" \
        --partition="$partition" \
        --gres=gpu:1 \
        --mem="$MEM" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --output="$LOG_DIR/${job_name}.out" \
        --error="$LOG_DIR/${job_name}.err" \
        --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'")
    JOB_IDS+=("$job_id")
done

echo ""
echo "Submitted jobs: ${JOB_IDS[*]}"
echo "Check status with: squeue -u $(whoami)"
