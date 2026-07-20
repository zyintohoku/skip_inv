#!/bin/bash

# Compute CFG/DDIM directional features for the 10 most seed-sensitive prompts.
#
# This reuses saved initial latents from outputs/aidi_gs7_seed*/init_latents.pt
# and records the four features from CFG_scale.pdf:
#   weighted_alignment_C_u
#   orthogonal_bending_ratio_rho_perp_u
#   reverse_direction_ratio_rho_minus_u
#   uncond_trajectory_deviation_auc_D_u
#
# Usage:
#   bash scripts/run_cfg_directional_features_seed_sensitive_top10.sh yagi35
#
# Environment overrides:
#   TOP_K=10
#   SAVED_SEEDS=1-10
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   OUTPUT_DIR=outputs/cfg_directional_features_seed_sensitive_top10
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   MEM=48G
#   CPUS_PER_TASK=4
#   SAVE_IMAGES=1
#   SLOTS_PER_NODE=auto
#   NODE_SLOT_COUNTS=6,4

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
mkdir -p log

TOP_K=${TOP_K:-10}
SAVED_SEEDS=${SAVED_SEEDS:-1-10}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/cfg_directional_features_seed_sensitive_top10}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SAVE_IMAGES=${SAVE_IMAGES:-1}
SLOTS_PER_NODE=${SLOTS_PER_NODE:-auto}
NODE_SLOT_COUNTS=${NODE_SLOT_COUNTS:-}

SOURCE_CSV=${SOURCE_CSV:-results/aidi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv}
PSNR_DETAIL_CSV=${PSNR_DETAIL_CSV:-results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv}

if [ $# -eq 0 ]; then
    echo "Error: No nodes specified."
    echo "Usage: bash scripts/run_cfg_directional_features_seed_sensitive_top10.sh <node1> [node2] ..."
    exit 1
fi

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

read_top_ids() {
    awk -F, -v top_k="$TOP_K" 'NR > 1 && NR <= top_k + 1 {print $1}' "$SOURCE_CSV"
}

mapfile -t SAMPLE_IDS < <(read_top_ids)

echo "Node slot plan:"
read -r -a NODE_SLOTS <<< "$(build_node_slots)"

if [ "${#SAMPLE_IDS[@]}" -gt "${#NODE_SLOTS[@]}" ]; then
    echo "Error: ${#SAMPLE_IDS[@]} jobs requested but only ${#NODE_SLOTS[@]} node slots configured/available."
    echo "Requested sample_ids: ${SAMPLE_IDS[*]}"
    echo "Expanded slot order: ${NODE_SLOTS[*]:-none}"
    echo "Add more nodes or set NODE_SLOT_COUNTS/SLOTS_PER_NODE to allow enough jobs."
    exit 1
fi

echo "============================================================"
echo "CFG Directional Features Seed-Sensitive Top-K Job Submission"
echo "============================================================"
echo "Top K: $TOP_K"
echo "Sample ids: ${SAMPLE_IDS[*]}"
echo "Saved init latent seeds: $SAVED_SEEDS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Source CSV: $SOURCE_CSV"
echo "PSNR detail CSV: $PSNR_DETAIL_CSV"
echo "Save images: $SAVE_IMAGES"
echo "Nodes:"
for i in "${!NODE_ARRAY[@]}"; do
    echo "  $i: ${NODE_ARRAY[$i]}"
done
echo "Expanded slot order: ${NODE_SLOTS[*]}"
echo ""

SAVE_IMAGE_FLAG=""
if [ "$SAVE_IMAGES" = "1" ]; then
    SAVE_IMAGE_FLAG="--save_images"
fi

for i in "${!SAMPLE_IDS[@]}"; do
    sample_id=${SAMPLE_IDS[$i]}
    selected_node=${NODE_SLOTS[$i]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" 2>/dev/null | head -1)
    if [ -z "$partition" ]; then
        echo "Warning: cannot determine partition for $selected_node, using 48-4"
        partition="48-4"
    fi

    job_name="cfg_dir_sensitive_s${sample_id}"
    echo "Submitting $job_name -> $selected_node (partition=$partition)"

    sbatch --job-name="$job_name" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="log/${job_name}.out" \
           --error="log/${job_name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python cfg_directional_features_saved_latents.py --sample_id $sample_id --mapping_file $MAPPING_FILE --source_csv $SOURCE_CSV --psnr_detail_csv $PSNR_DETAIL_CSV --top_k $TOP_K --output $OUTPUT_DIR --model_name $MODEL_NAME --guidance_scale $GUIDANCE_SCALE --num_of_ddim_steps $NUM_DDIM_STEPS --seeds $SAVED_SEEDS --device cuda $SAVE_IMAGE_FLAG'"
done

echo ""
echo "Submitted ${#SAMPLE_IDS[@]} sample job(s)."
echo "Each job writes its own sample_XXXX/per_seed_cfg_directional_features.csv."
echo "Check status: squeue -u \$(whoami)"
