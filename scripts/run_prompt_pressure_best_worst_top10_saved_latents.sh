#!/bin/bash

# Submit prompt-pressure tracing jobs for best/worst top10 prompts using saved
# init latents from outputs/aidi_gs7_seed*/init_latents.pt.
#
# This intentionally does not reuse outputs/prompt_pressure_top10 because that
# directory was generated from freshly sampled init latents. The traces produced
# by this script are aligned with results/aidi_gs7_seed_psnr/*.csv through
# (sample_id, seed).
#
# Usage:
#   bash scripts/run_prompt_pressure_best_worst_top10_saved_latents.sh yagi35
#   bash scripts/run_prompt_pressure_best_worst_top10_saved_latents.sh yagi35 yagi38 yagi39
#
# Environment overrides:
#   TOP_SET=both|best|worst
#   TOP_K=10
#   SAVED_SEEDS=1-10
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   OUTPUT_DIR=outputs/prompt_pressure_best_worst_top10_saved_latents
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

TOP_SET=${TOP_SET:-both}
TOP_K=${TOP_K:-10}
SAVED_SEEDS=${SAVED_SEEDS:-1-10}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/prompt_pressure_best_worst_top10_saved_latents}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

BEST_CSV="results/aidi_gs7_seed_psnr/prompt_psnr_best30.csv"
WORST_CSV="results/aidi_gs7_seed_psnr/prompt_psnr_worst30.csv"

if [ "$TOP_SET" != "both" ] && [ "$TOP_SET" != "best" ] && [ "$TOP_SET" != "worst" ]; then
    echo "Error: TOP_SET must be one of: both, best, worst"
    exit 1
fi

slurm_load_node_slots NODE_ARRAY "$@"

read_top_ids() {
    local csv_path=$1
    awk -F, -v top_k="$TOP_K" 'NR > 1 && NR <= top_k + 1 {print $1}' "$csv_path"
}

JOB_LABELS=()
JOB_IDS=()

if [ "$TOP_SET" = "both" ] || [ "$TOP_SET" = "best" ]; then
    while IFS= read -r sample_id; do
        JOB_LABELS+=("best")
        JOB_IDS+=("$sample_id")
    done < <(read_top_ids "$BEST_CSV")
fi

if [ "$TOP_SET" = "both" ] || [ "$TOP_SET" = "worst" ]; then
    while IFS= read -r sample_id; do
        JOB_LABELS+=("worst")
        JOB_IDS+=("$sample_id")
    done < <(read_top_ids "$WORST_CSV")
fi

echo "============================================================"
echo "Prompt Pressure Best/Worst Top-K Saved-Latent Job Submission"
echo "============================================================"
echo "Top set: $TOP_SET"
echo "Top K per set: $TOP_K"
echo "Jobs: ${#JOB_IDS[@]} sample_ids"
echo "Saved init latent seeds: $SAVED_SEEDS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo "Output: $OUTPUT_DIR"
echo "Nodes:"
for i in "${!NODE_ARRAY[@]}"; do
    echo "  $i: ${NODE_ARRAY[$i]}"
done
echo ""

for i in "${!JOB_IDS[@]}"; do
    label=${JOB_LABELS[$i]}
    sample_id=${JOB_IDS[$i]}
    selected_node=$(slurm_node_for_index "$i" "${NODE_ARRAY[@]}")
    partition=$(slurm_partition "$selected_node")

    job_name="pp_saved_${label}_s${sample_id}"

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
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python prompt_pressure_saved_latents.py --sample_id $sample_id --label $label --mapping_file $MAPPING_FILE --output $OUTPUT_DIR --model_name $MODEL_NAME --guidance_scale $GUIDANCE_SCALE --num_of_ddim_steps $NUM_DDIM_STEPS --seeds $SAVED_SEEDS --source_csv results/aidi_gs7_seed_psnr/prompt_psnr_${label}30.csv --top_k $TOP_K --device cuda'"
done

echo ""
echo "All ${#JOB_IDS[@]} saved-latent best/worst prompt-pressure jobs submitted."
echo "Check status: squeue -u \$(whoami)"
echo "Logs: tail -f log/pp_saved_*.out"
