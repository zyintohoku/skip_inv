#!/bin/bash

# Submit run.py FPI gen/inv/rec tests for prompt 0 with seeds 0 and 1.
#
# Usage:
#   bash scripts/run_runpy_seed0_1_fpi_sample0.sh
#   bash scripts/run_runpy_seed0_1_fpi_sample0.sh yagi35
#
# Useful overrides:
#   SAMPLE_IDS=0 METHOD=fpi bash scripts/run_runpy_seed0_1_fpi_sample0.sh yagi35
#   OUTPUT_PREFIX=outputs/runpy_fpi_sample0_seed bash scripts/run_runpy_seed0_1_fpi_sample0.sh

PROJECT_DIR=${PROJECT_DIR:-$PWD}

TARGET_NODES="yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"
SEEDS=(0 1)

CONDA_ENV=${CONDA_ENV:-afpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
SAMPLE_IDS=${SAMPLE_IDS:-0}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-1.0}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
SOURCE_INIT_PREFIX=${SOURCE_INIT_PREFIX:-outputs/aidi_gs7_seed}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/runpy_${METHOD}_sample${SAMPLE_IDS}_seed}
LOG_DIR=${LOG_DIR:-log}

mkdir -p "$LOG_DIR"

if [ -z "$1" ]; then
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

    selected_node=${NODE_ARRAY[0]}
else
    selected_node=$1
    echo "Using specified node: $selected_node"
fi

partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
if [ -z "$partition" ]; then
    echo "Cannot determine partition for $selected_node, using default 48-4"
    partition="48-4"
fi

echo "Project dir: $PROJECT_DIR"
echo "Node: $selected_node (partition: $partition)"
echo "Seeds: ${SEEDS[*]}"
echo "Method: $METHOD"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Sample ids: $SAMPLE_IDS"
echo "Source init prefix: $SOURCE_INIT_PREFIX"
echo "Output prefix: $OUTPUT_PREFIX"
echo ""

for seed in "${SEEDS[@]}"; do
    name="runpy_${METHOD}_sample${SAMPLE_IDS}_s${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}"
    script_file="${LOG_DIR}/${name}.slurm.sh"

    echo "  Submitting seed=$seed -> $selected_node (output=$output_dir)"

    cat > "$script_file" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --nodelist=$selected_node
#SBATCH --partition=$partition
###########RESOURCES###########
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
###############################
#SBATCH --output=${LOG_DIR}/${name}.out
#SBATCH --error=${LOG_DIR}/${name}.err
#SBATCH -v

cd "$PROJECT_DIR"

echo "hostname=\$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
##############################
python -c "import torch; print('torch_cuda_available=' + str(torch.cuda.is_available())); raise SystemExit(0 if torch.cuda.is_available() else 1)"

if [ ! -f "${SOURCE_INIT_PREFIX}${seed}/init_latents.pt" ]; then
    echo "Missing source init latent: ${SOURCE_INIT_PREFIX}${seed}/init_latents.pt" >&2
    exit 1
fi

python run.py \\
    --method $METHOD \\
    --guidance_scale $GUIDANCE_SCALE \\
    --seed $seed \\
    --output $output_dir \\
    --num_of_ddim_steps $NUM_DDIM_STEPS \\
    --delta_threshold $DELTA_THRESHOLD \\
    --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD \\
    --mapping_file $MAPPING_FILE \\
    --sample_ids $SAMPLE_IDS \\
    --model_name $MODEL_NAME \\
    --source_init_prefix $SOURCE_INIT_PREFIX
SCRIPT_EOF

    chmod +x "$script_file"
    sbatch "$script_file"
done

echo ""
echo "All ${#SEEDS[@]} jobs submitted."
echo "Check status with: squeue -u \$(whoami)"
