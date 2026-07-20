#!/bin/bash

# Submit prompt-pressure tracing jobs for the 10 most seed-sensitive prompts.
#
# This uses the saved init latents from outputs/aidi_gs7_seed*/init_latents.pt,
# so each sample is traced with the exact initial latents behind the
# aidi_gs7_seed_psnr result.
#
# Usage:
#   bash scripts/run_prompt_pressure_seed_sensitive_top10.sh yagi35
#   bash scripts/run_prompt_pressure_seed_sensitive_top10.sh yagi35 yagi38 yagi39
#
# Environment overrides:
#   TOP_K=10
#   SAVED_SEEDS=1-10
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   OUTPUT_DIR=outputs/prompt_pressure_seed_sensitive_top10
#   MAPPING_FILE=PIE_bench/mapping_file.json
#   MODEL_NAME=CompVis/stable-diffusion-v1-4
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:2,yagi38:1
#   MEM=48G
#   CPUS_PER_TASK=4

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TOP_K=${TOP_K:-10}
SAVED_SEEDS=${SAVED_SEEDS:-1-10}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/prompt_pressure_seed_sensitive_top10}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

SOURCE_CSV="results/aidi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv"

slurm_load_node_slots NODE_ARRAY "$@"

read_top_ids() {
    awk -F, -v top_k="$TOP_K" 'NR > 1 && NR <= top_k + 1 {print $1}' "$SOURCE_CSV"
}

mapfile -t SAMPLE_IDS < <(read_top_ids)

echo "============================================================"
echo "Prompt Pressure Seed-Sensitive Top-K Job Submission"
echo "============================================================"
echo "Top K: $TOP_K"
echo "Jobs: ${#SAMPLE_IDS[@]} sample_ids"
echo "Saved init latent seeds: $SAVED_SEEDS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Nodes:"
for i in "${!NODE_ARRAY[@]}"; do
    echo "  $i: ${NODE_ARRAY[$i]}"
done
echo ""

for i in "${!SAMPLE_IDS[@]}"; do
    sample_id=${SAMPLE_IDS[$i]}
    selected_node=$(slurm_node_for_index "$i" "${NODE_ARRAY[@]}")
    partition=$(slurm_partition "$selected_node")

    job_name="pp_seed_sensitive_s${sample_id}"

    echo "Submitting $job_name -> $selected_node (partition=$partition)"

    sbatch --job-name="$job_name" \
           --chdir="$PROJECT_DIR" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="$LOG_DIR/${job_name}.out" \
           --error="$LOG_DIR/${job_name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python prompt_pressure_saved_latents.py --sample_id $sample_id --mapping_file $MAPPING_FILE --output $OUTPUT_DIR --model_name $MODEL_NAME --guidance_scale $GUIDANCE_SCALE --num_of_ddim_steps $NUM_DDIM_STEPS --seeds $SAVED_SEEDS --device cuda'"
done

echo ""
echo "All ${#SAMPLE_IDS[@]} seed-sensitive prompt-pressure jobs submitted."
echo "Check status: squeue -u \$(whoami)"
echo "Logs: tail -f log/pp_seed_sensitive_*.out"
