#!/bin/bash

# Submit FPI gen-inv-rec jobs for puppy/dog field prompts over seeds 1..10.
#
# Usage:
#   bash scripts/run_puppy_dog_field_fpi_gen_inv_rec.sh
#   bash scripts/run_puppy_dog_field_fpi_gen_inv_rec.sh yagi35 yagi38

PROJECT_DIR=${PROJECT_DIR:-$PWD}

TARGET_NODES=${TARGET_NODES:-"yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"}
SEEDS=($(seq 1 10))
PROMPT_CSV=${PROMPT_CSV:-results/fpi_gs7_seed_psnr/prompt_structure_analysis/puppy_dog_field_prompts.csv}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
METHOD=${METHOD:-fpi}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/puppy_dog_field_fpi_gs7_seed}
RESULTS_DIR=${RESULTS_DIR:-results/fpi_gs7_seed_psnr/puppy_dog_field_fpi}
RESULT_NAME=${RESULT_NAME:-puppy_dog_field_fpi}
CONDA_ENV=${CONDA_ENV:-afpi}
JOBS_PER_NODE=${JOBS_PER_NODE:-1}
RUN_AGGREGATE=${RUN_AGGREGATE:-1}

AGG_OUTPUTS_DIR=$(dirname "$OUTPUT_PREFIX")
AGG_RUN_PREFIX=$(basename "$OUTPUT_PREFIX")
mkdir -p log

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

NODE_SLOTS=()
for node in "${NODE_ARRAY[@]}"; do
    for ((slot=0; slot<JOBS_PER_NODE; slot++)); do
        NODE_SLOTS+=("$node")
    done
done

echo "Total jobs: ${#SEEDS[@]}"
echo "Prompt CSV: $PROMPT_CSV"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Results dir: $RESULTS_DIR"
echo ""

JOB_IDS=()
job_idx=0
for seed in "${SEEDS[@]}"; do
    selected_node=${NODE_SLOTS[$((job_idx % ${#NODE_SLOTS[@]}))]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    if [ -z "$partition" ]; then
        partition="48-4"
    fi

    name="pupdog_field_fpi_s${seed}"
    output_dir="${OUTPUT_PREFIX}${seed}"
    echo "Submitting seed=$seed -> $selected_node (partition=$partition, output=$output_dir)"

    job_id=$(sbatch --parsable \
        --job-name="$name" \
        --nodelist="$selected_node" \
        --partition="$partition" \
        --gres=gpu:1 \
        --mem=48G \
        --cpus-per-task=4 \
        --output="log/${name}.out" \
        --error="log/${name}.err" \
        --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python run_prompt_ablation_fpi_gen_inv_rec.py --prompt_csv $PROMPT_CSV --method $METHOD --guidance_scale $GUIDANCE_SCALE --seed $seed --output $output_dir --num_of_ddim_steps $NUM_DDIM_STEPS --delta_threshold $DELTA_THRESHOLD --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD'")
    JOB_IDS+=("$job_id")
    job_idx=$((job_idx + 1))
done

if [ "$RUN_AGGREGATE" = "1" ]; then
    dependency=$(IFS=:; echo "${JOB_IDS[*]}")
    agg_name="pupdog_field_fpi_agg"
    agg_job_id=$(sbatch --parsable \
        --job-name="$agg_name" \
        --dependency="afterok:$dependency" \
        --mem=8G \
        --cpus-per-task=2 \
        --output="log/${agg_name}.out" \
        --error="log/${agg_name}.err" \
        --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python analysis/compute_prompt_grid_fpi_psnr.py --outputs_dir $AGG_OUTPUTS_DIR --run_prefix $AGG_RUN_PREFIX --results_dir $RESULTS_DIR --name $RESULT_NAME --seeds 1-10'")
    echo "Submitted aggregate job: $agg_job_id"
fi

echo "Submitted seed jobs: ${JOB_IDS[*]}"
echo "Check status with: squeue -u $(whoami)"
