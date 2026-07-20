#!/bin/bash

# Re-run AIDI GS=7 generation only for saved seed init latents and record
# generation-time prompt-pressure metrics P_t/R_t. This does not run inversion
# or reconstruction.
#
# Usage:
#   bash scripts/run_aidi_gs7_seed_generation_pressure.sh yagi35
#   bash scripts/run_aidi_gs7_seed_generation_pressure.sh yagi35 yagi38 yagi39
#
# Environment overrides:
#   SEEDS=1-10
#   SAMPLE_IDS=0-699
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   OUTPUT_DIR=outputs/aidi_gs7_seed_generation_pressure
#   MAPPING_FILE=PIE_bench/mapping_file.json
#   MODEL_NAME=CompVis/stable-diffusion-v1-4
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   MEM=32G
#   CPUS_PER_TASK=4
#   SAVE_IMAGE_SAMPLE_IDS=0-4
#   SLOTS_PER_NODE=auto
#   NODE_SLOT_COUNTS=6,4

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
mkdir -p log

SEEDS=${SEEDS:-1-10}
SAMPLE_IDS=${SAMPLE_IDS:-0-699}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/aidi_gs7_seed_generation_pressure}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-32G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SAVE_IMAGE_SAMPLE_IDS=${SAVE_IMAGE_SAMPLE_IDS:-}
SLOTS_PER_NODE=${SLOTS_PER_NODE:-auto}
NODE_SLOT_COUNTS=${NODE_SLOT_COUNTS:-}

if [ $# -eq 0 ]; then
    echo "Error: No nodes specified."
    echo "Usage: bash scripts/run_aidi_gs7_seed_generation_pressure.sh <node1> [node2] ..."
    exit 1
fi

parse_seed_spec() {
    local spec=$1
    local out=()
    local token start end value
    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        token=${token//[[:space:]]/}
        if [ -z "$token" ]; then
            continue
        fi
        if [[ "$token" == *-* ]]; then
            start=${token%-*}
            end=${token#*-}
            for ((value=start; value<=end; value++)); do
                out+=("$value")
            done
        else
            out+=("$token")
        fi
    done
    echo "${out[*]}"
}

read -r -a SEED_ARRAY <<< "$(parse_seed_spec "$SEEDS")"
NODE_ARRAY=("$@")

available_gpu_slots() {
    local node=$1
    local node_info gres total used
    node_info=$(scontrol show node "$node" 2>/dev/null || true)
    if [ -z "$node_info" ]; then
        echo 1
        return
    fi

    gres=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="Gres"{print $2; exit}')
    total=$(printf '%s\n' "$gres" | sed -n 's/.*gpu[^:]*:\([0-9][0-9]*\).*/\1/p')
    if [ -z "$total" ]; then
        total=$(printf '%s\n' "$gres" | sed -n 's/.*gpu:\([0-9][0-9]*\).*/\1/p')
    fi

    used=$(printf '%s\n' "$node_info" | tr ' ' '\n' | awk -F= '$1=="AllocTRES"{print $2; exit}' | sed -n 's/.*gres\/gpu=\([0-9][0-9]*\).*/\1/p')
    total=${total:-1}
    used=${used:-0}
    if [ "$total" -le "$used" ]; then
        echo 0
    else
        echo $((total - used))
    fi
}

build_node_slots() {
    local slots=()
    local node n slot node_idx count_text
    local count_array=()
    if [ -n "$NODE_SLOT_COUNTS" ]; then
        IFS=',' read -ra count_array <<< "$NODE_SLOT_COUNTS"
        if [ "${#count_array[@]}" -ne "${#NODE_ARRAY[@]}" ]; then
            echo "Error: NODE_SLOT_COUNTS must have one count per node." >&2
            echo "Example: NODE_SLOT_COUNTS=6,4 bash $0 yagi35 yagi38" >&2
            exit 1
        fi
    fi

    for node_idx in "${!NODE_ARRAY[@]}"; do
        node=${NODE_ARRAY[$node_idx]}
        if [ -n "$NODE_SLOT_COUNTS" ]; then
            count_text=${count_array[$node_idx]}
            count_text=${count_text//[[:space:]]/}
            n=$count_text
        elif [ "$SLOTS_PER_NODE" = "auto" ]; then
            n=$(available_gpu_slots "$node")
        else
            n=$SLOTS_PER_NODE
        fi
        if [ "$n" -lt 1 ]; then
            echo "  $node: 0 available slots" >&2
            continue
        fi
        echo "  $node: $n slot(s)" >&2
        for ((slot=0; slot<n; slot++)); do
            slots+=("$node")
        done
    done
    if [ "${#slots[@]}" -eq 0 ]; then
        echo "Error: no available node slots found." >&2
        exit 1
    fi
    echo "${slots[*]}"
}

echo "Node slot plan:"
read -r -a NODE_SLOTS <<< "$(build_node_slots)"

if [ "${#SEED_ARRAY[@]}" -gt "${#NODE_SLOTS[@]}" ]; then
    echo "Error: ${#SEED_ARRAY[@]} jobs requested but only ${#NODE_SLOTS[@]} node slots configured/available."
    echo "Requested seeds: ${SEED_ARRAY[*]}"
    echo "Expanded slot order: ${NODE_SLOTS[*]:-none}"
    echo "Add more nodes or set NODE_SLOT_COUNTS/SLOTS_PER_NODE to allow enough jobs."
    exit 1
fi

echo "============================================================"
echo "AIDI GS=7 Generation Pressure Job Submission"
echo "============================================================"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Sample ids: $SAMPLE_IDS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Save image sample ids: ${SAVE_IMAGE_SAMPLE_IDS:-none}"
echo "Nodes:"
for i in "${!NODE_ARRAY[@]}"; do
    echo "  $i: ${NODE_ARRAY[$i]}"
done
echo "Expanded slot order: ${NODE_SLOTS[*]}"
echo ""

for i in "${!SEED_ARRAY[@]}"; do
    seed=${SEED_ARRAY[$i]}
    if [ "$i" -lt "${#NODE_SLOTS[@]}" ]; then
        selected_node=${NODE_SLOTS[$i]}
    fi
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" 2>/dev/null | head -1)
    if [ -z "$partition" ]; then
        echo "Warning: cannot determine partition for $selected_node, using 48-4"
        partition="48-4"
    fi

    job_name="gen_pressure_gs7_s${seed}"
    init_latents_path="outputs/aidi_gs7_seed${seed}/init_latents.pt"
    save_images_arg=""
    if [ -n "$SAVE_IMAGE_SAMPLE_IDS" ]; then
        save_images_arg="--save_image_sample_ids $SAVE_IMAGE_SAMPLE_IDS"
    fi

    echo "Submitting $job_name -> $selected_node (partition=$partition)"

    sbatch --job-name="$job_name" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="log/${job_name}.out" \
           --error="log/${job_name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python generate_aidi_gs7_seed_pressure.py --seed $seed --sample_ids $SAMPLE_IDS --mapping_file $MAPPING_FILE --init_latents_path $init_latents_path --output $OUTPUT_DIR --model_name $MODEL_NAME --guidance_scale $GUIDANCE_SCALE --num_of_ddim_steps $NUM_DDIM_STEPS $save_images_arg --device cuda'"
done

echo ""
echo "All ${#SEED_ARRAY[@]} generation-pressure jobs submitted."
echo "Check status: squeue -u \$(whoami)"
echo "Logs: tail -f log/gen_pressure_gs7_s*.out"
