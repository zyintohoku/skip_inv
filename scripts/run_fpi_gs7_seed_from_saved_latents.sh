#!/bin/bash

# Submit FPI inversion/reconstruction jobs from saved AIDI generated latents.
# This does not regenerate source images; it loads outputs/aidi_gs7_seed*/gen_latents.pt.
#
# Usage:
#   bash scripts/run_fpi_gs7_seed_from_saved_latents.sh
#   bash scripts/run_fpi_gs7_seed_from_saved_latents.sh yagi35
#   bash scripts/run_fpi_gs7_seed_from_saved_latents.sh yagi35 yagi38 yagi39
#
# Node job limits:
#   JOBS_PER_NODE=2 bash scripts/run_fpi_gs7_seed_from_saved_latents.sh yagi35 yagi38
#   NODE_SLOT_COUNTS=2,1 bash scripts/run_fpi_gs7_seed_from_saved_latents.sh yagi35 yagi38

PROJECT_DIR=${PROJECT_DIR:-$PWD}

TARGET_NODES="yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"
SEEDS=($(seq 1 10))
GUIDANCE_SCALE=7
METHOD=fpi
SAMPLE_IDS=${SAMPLE_IDS:-0-699}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
SOURCE_PREFIX=${SOURCE_PREFIX:-outputs/aidi_gs7_seed}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/fpi_gs7_seed}
JOBS_PER_NODE=${JOBS_PER_NODE:-1}
NODE_SLOT_COUNTS=${NODE_SLOT_COUNTS:-}

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

    echo "Found ${#NODE_ARRAY[@]} available nodes:"
    printf '%s\n' "${NODE_ARRAY[@]}"
    echo ""
else
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  Job $i: ${NODE_ARRAY[$i]}"
    done
    echo ""
fi

build_node_slots() {
    local slots=()
    local count_array=()
    local node_idx node count_text n slot

    if [ -n "$NODE_SLOT_COUNTS" ]; then
        IFS=',' read -ra count_array <<< "$NODE_SLOT_COUNTS"
        if [ "${#count_array[@]}" -ne "${#NODE_ARRAY[@]}" ]; then
            echo "Error: NODE_SLOT_COUNTS must have one count per node." >&2
            echo "Example: NODE_SLOT_COUNTS=2,1 bash $0 yagi35 yagi38" >&2
            exit 1
        fi
    fi

    for node_idx in "${!NODE_ARRAY[@]}"; do
        node=${NODE_ARRAY[$node_idx]}
        if [ -n "$NODE_SLOT_COUNTS" ]; then
            count_text=${count_array[$node_idx]}
            count_text=${count_text//[[:space:]]/}
            n=$count_text
        else
            n=$JOBS_PER_NODE
        fi

        if ! [[ "$n" =~ ^[0-9]+$ ]]; then
            echo "Error: job count for $node must be a non-negative integer, got '$n'." >&2
            exit 1
        fi
        if [ "$n" -lt 1 ]; then
            echo "  $node: 0 job slot(s)" >&2
            continue
        fi

        echo "  $node: $n job slot(s)" >&2
        for ((slot=0; slot<n; slot++)); do
            slots+=("$node")
        done
    done

    if [ "${#slots[@]}" -eq 0 ]; then
        echo "Error: no node job slots configured." >&2
        exit 1
    fi
    echo "${slots[*]}"
}

read -r -a NODE_SLOTS <<< "$(build_node_slots)"

echo "Total jobs: ${#SEEDS[@]}"
echo "Nodes: ${#NODE_ARRAY[@]}"
echo "Node job slots in rotation: ${#NODE_SLOTS[@]}"
echo "method=$METHOD guidance_scale=$GUIDANCE_SCALE sample_ids=$SAMPLE_IDS"
echo ""

job_idx=0
for seed in "${SEEDS[@]}"; do
    NODE_IDX=$((job_idx % ${#NODE_SLOTS[@]}))
    selected_node=${NODE_SLOTS[$NODE_IDX]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    if [ -z "$partition" ]; then
        echo "  Cannot determine partition for $selected_node, using default 48-4"
        partition="48-4"
    fi

    name="fpi_gs${GUIDANCE_SCALE}_saved_s${seed}"
    source_dir="${SOURCE_PREFIX}${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}_from_saved_latents"

    echo "  Submitting seed=$seed -> $selected_node (partition=$partition, source=$source_dir)"

    sbatch --job-name="$name" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --output="log/${name}.out" \
           --error="log/${name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python run_fpi_from_saved_latents.py --method $METHOD --guidance_scale $GUIDANCE_SCALE --seed $seed --source_dir $source_dir --output $output_dir --mapping_file $MAPPING_FILE --sample_ids $SAMPLE_IDS --num_of_ddim_steps $NUM_DDIM_STEPS --delta_threshold $DELTA_THRESHOLD --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD'"

    job_idx=$((job_idx + 1))
done

echo ""
echo "All ${#SEEDS[@]} jobs submitted."
echo "Check status with: squeue -u \$(whoami)"
