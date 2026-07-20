#!/bin/bash

# Submit additive-latent perturbation jobs.
#
# Each job reuses saved initial latents from outputs/aidi_gs7_seed*/init_latents.pt,
# creates perturbed latents z_T^(i) = z_T + sigma * xi_i, denoises them with
# the same prompt/CFG scale, saves perturbed generated images, and records terminal sensitivity:
#   S0 = ||z0^(i) - z0|| / ||zT^(i) - zT||
#
# Usage:
#   bash scripts/run_latent_perturbation_seed_sensitive_top10.sh yagi35
#   bash scripts/run_latent_perturbation_seed_sensitive_top10.sh yagi35 yagi38
#
# Environment overrides:
#   SAMPLE_IDS=71
#   TOP_K=1
#   SAVED_SEEDS=1-10
#   NOISE_SIGMA=0.01
#   NUM_PERTURBATIONS=10
#   PERTURBATION_SEED=20260503
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   OUTPUT_DIR=outputs/latent_perturbation_best_top1_sigma001
#   SAVE_BASELINE_IMAGES=1
#   SAVE_PERTURB_IMAGES=1
#   MAPPING_FILE=PIE_bench/mapping_file.json
#   MODEL_NAME=CompVis/stable-diffusion-v1-4
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   MEM=32G
#   CPUS_PER_TASK=4
#   SLOTS_PER_NODE=auto
#   NODE_SLOT_COUNTS=6,4

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
mkdir -p log

TOP_K=${TOP_K:-1}
SAMPLE_IDS=${SAMPLE_IDS:-}
SAVED_SEEDS=${SAVED_SEEDS:-1-10}
NOISE_SIGMA=${NOISE_SIGMA:-0.01}
NUM_PERTURBATIONS=${NUM_PERTURBATIONS:-10}
PERTURBATION_SEED=${PERTURBATION_SEED:-20260503}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/latent_perturbation_best_top1_sigma001}
SAVE_BASELINE_IMAGES=${SAVE_BASELINE_IMAGES:-1}
SAVE_PERTURB_IMAGES=${SAVE_PERTURB_IMAGES:-1}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-32G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SLOTS_PER_NODE=${SLOTS_PER_NODE:-auto}
NODE_SLOT_COUNTS=${NODE_SLOT_COUNTS:-}

SOURCE_CSV=${SOURCE_CSV:-results/aidi_gs7_seed_psnr/prompt_psnr_best30.csv}
PSNR_DETAIL_CSV=${PSNR_DETAIL_CSV:-results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv}
LABEL=${LABEL:-best}

if [ $# -eq 0 ]; then
    echo "Error: No nodes specified."
    echo "Usage: bash scripts/run_latent_perturbation_seed_sensitive_top10.sh <node1> [node2] ..."
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

expand_sample_ids() {
    local spec=$1
    local token
    local ids=()
    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        token=${token//[[:space:]]/}
        [ -n "$token" ] && ids+=("$token")
    done
    if [ "${#ids[@]}" -eq 0 ]; then
        echo "Error: SAMPLE_IDS was set but no sample ids were parsed: $spec" >&2
        exit 1
    fi
    printf '%s\n' "${ids[@]}"
}

expand_seed_spec() {
    local spec=$1
    local token range_part stride_part start end stride seed
    local seeds=()
    IFS=',' read -ra tokens <<< "$spec"
    for token in "${tokens[@]}"; do
        token=${token//[[:space:]]/}
        [ -z "$token" ] && continue
        if [[ "$token" != *-* ]]; then
            seeds+=("$token")
            continue
        fi
        range_part=$token
        stride_part=1
        if [[ "$token" == *:* ]]; then
            range_part=${token%%:*}
            stride_part=${token##*:}
        fi
        start=${range_part%%-*}
        end=${range_part##*-}
        stride=$stride_part
        for ((seed=start; seed<=end; seed+=stride)); do
            seeds+=("$seed")
        done
    done
    if [ "${#seeds[@]}" -eq 0 ]; then
        echo "Error: no seeds parsed from SAVED_SEEDS=$spec" >&2
        exit 1
    fi
    echo "${seeds[*]}"
}

if [ -n "$SAMPLE_IDS" ]; then
    mapfile -t SAMPLE_ID_LIST < <(expand_sample_ids "$SAMPLE_IDS")
else
    mapfile -t SAMPLE_ID_LIST < <(read_top_ids)
fi
read -r -a SEED_IDS <<< "$(expand_seed_spec "$SAVED_SEEDS")"

JOB_SAMPLE_IDS=()
JOB_SEED_IDS=()
for sample_id in "${SAMPLE_ID_LIST[@]}"; do
    for seed in "${SEED_IDS[@]}"; do
        JOB_SAMPLE_IDS+=("$sample_id")
        JOB_SEED_IDS+=("$seed")
    done
done

echo "Node slot plan:"
read -r -a NODE_SLOTS <<< "$(build_node_slots)"

if [ "${#JOB_SAMPLE_IDS[@]}" -gt "${#NODE_SLOTS[@]}" ]; then
    echo "Error: ${#JOB_SAMPLE_IDS[@]} seed jobs requested but only ${#NODE_SLOTS[@]} node slots configured/available."
    echo "Requested sample_ids: ${SAMPLE_ID_LIST[*]}"
    echo "Requested seed ids: ${SEED_IDS[*]}"
    echo "Expanded slot order: ${NODE_SLOTS[*]:-none}"
    echo "Add more nodes or set NODE_SLOT_COUNTS/SLOTS_PER_NODE to allow enough jobs."
    exit 1
fi

echo "============================================================"
echo "Additive Latent Perturbation Job Submission"
echo "============================================================"
echo "Top K: $TOP_K"
echo "Sample ids: ${SAMPLE_ID_LIST[*]}"
echo "Seed ids: ${SEED_IDS[*]}"
echo "Jobs: ${#JOB_SAMPLE_IDS[@]} sample-seed job(s)"
echo "Saved init latent seeds: $SAVED_SEEDS"
echo "Noise sigma: $NOISE_SIGMA"
echo "perturbations per seed: $NUM_PERTURBATIONS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Source CSV: $SOURCE_CSV"
echo "Label: $LABEL"
echo "Save baseline images: $SAVE_BASELINE_IMAGES"
echo "Save perturb images: $SAVE_PERTURB_IMAGES"
echo "Nodes:"
for i in "${!NODE_ARRAY[@]}"; do
    echo "  $i: ${NODE_ARRAY[$i]}"
done
echo "Expanded slot order: ${NODE_SLOTS[*]}"
echo ""

BASELINE_IMAGE_ARG=""
if [ "$SAVE_BASELINE_IMAGES" = "1" ]; then
    BASELINE_IMAGE_ARG="--save_baseline_images"
fi

PERTURB_IMAGE_ARG=""
if [ "$SAVE_PERTURB_IMAGES" = "1" ]; then
    PERTURB_IMAGE_ARG="--save_perturb_images"
fi

for i in "${!JOB_SAMPLE_IDS[@]}"; do
    sample_id=${JOB_SAMPLE_IDS[$i]}
    seed=${JOB_SEED_IDS[$i]}
    selected_node=${NODE_SLOTS[$i]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" 2>/dev/null | head -1)
    if [ -z "$partition" ]; then
        echo "Warning: cannot determine partition for $selected_node, using 48-4"
        partition="48-4"
    fi

    job_name="latent_perturb_${LABEL}_s${sample_id}_seed${seed}"

    echo "Submitting $job_name -> $selected_node (partition=$partition)"

    sbatch --job-name="$job_name" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="log/${job_name}.out" \
           --error="log/${job_name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python latent_perturbation_saved_latents.py --sample_id $sample_id --mapping_file $MAPPING_FILE --source_csv $SOURCE_CSV --psnr_detail_csv $PSNR_DETAIL_CSV --output $OUTPUT_DIR --label $LABEL --model_name $MODEL_NAME --guidance_scale $GUIDANCE_SCALE --num_of_ddim_steps $NUM_DDIM_STEPS --seeds $seed --noise_sigma $NOISE_SIGMA --num_perturbations $NUM_PERTURBATIONS --perturbation_seed $PERTURBATION_SEED $BASELINE_IMAGE_ARG $PERTURB_IMAGE_ARG --skip_sample_aggregates --device cuda'"
done

echo ""
echo "Submitted ${#JOB_SAMPLE_IDS[@]} sample-seed job(s)."
echo "Each job writes one seed_XXXXXX directory with baseline_generated.png and perturb_XXX_generated.png files."
echo "After jobs finish, run:"
echo "  python analysis/analyze_latent_perturbation_sensitivity.py --input_root $OUTPUT_DIR"
