#!/bin/bash

# Recompute prompt-pressure statistics/plots for best/worst top10 traces that
# were generated from saved aidi_gs7 seed init latents.
#
# Usage:
#   bash scripts/analyze_prompt_pressure_best_worst_top10_saved_latents.sh
#   bash scripts/analyze_prompt_pressure_best_worst_top10_saved_latents.sh --submit
#   bash scripts/analyze_prompt_pressure_best_worst_top10_saved_latents.sh --submit yagi35
#
# Environment overrides:
#   PROJECT_DIR=$PWD
#   INPUT_ROOT=outputs/prompt_pressure_best_worst_top10_saved_latents
#   OUTPUT_ROOT=results/prompt_pressure_best_worst_top10_saved_latents_analysis
#   CONDA_ENV=afpi
#   MEM=16G
#   CPUS_PER_TASK=4
#   TIME=02:00:00

set -euo pipefail

SUBMIT=false
if [ "${1:-}" = "--submit" ]; then
    SUBMIT=true
    shift
fi

#PROJECT_DIR=${PROJECT_DIR:-$PWD}
PROJECT_DIR=${PROJECT_DIR:-$PWD}
mkdir -p log

INPUT_ROOT=${INPUT_ROOT:-outputs/prompt_pressure_best_worst_top10_saved_latents}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/prompt_pressure_best_worst_top10_saved_latents_analysis}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-16G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
TIME=${TIME:-02:00:00}

if [ "$SUBMIT" = true ]; then
    job_name="analyze_pp_best_worst_saved_top10"
    sbatch_args=(
        --job-name="$job_name"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --time="$TIME"
        --output="log/${job_name}_%j.out"
        --error="log/${job_name}_%j.err"
    )

    if [ $# -ge 1 ]; then
        selected_node=$1
        partition=$(sinfo -N -h -o "%P" -n "$selected_node" 2>/dev/null | head -1)
        if [ -z "$partition" ]; then
            echo "Warning: cannot determine partition for $selected_node, using 48-4"
            partition="48-4"
        fi
        sbatch_args+=(--nodelist="$selected_node" --partition="$partition")
    fi

    echo "Submitting $job_name"
    echo "Project: $PROJECT_DIR"
    echo "Input: $INPUT_ROOT"
    echo "Output: $OUTPUT_ROOT"

    sbatch "${sbatch_args[@]}" \
        --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && PROJECT_DIR=\"$PROJECT_DIR\" INPUT_ROOT=\"$INPUT_ROOT\" OUTPUT_ROOT=\"$OUTPUT_ROOT\" bash scripts/analyze_prompt_pressure_best_worst_top10_saved_latents.sh'"
    exit 0
fi

python analysis/compute_prompt_pressure_distribution_metrics.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/distribution_metrics"

python analysis/plot_prompt_pressure_by_seed.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/plots_by_sample"

python analysis/plot_prompt_pressure_normalized.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/plots_normalized" \
    --normalization max

python analysis/plot_prompt_pressure_normalized.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/plots_normalized" \
    --normalization sum

python analysis/plot_guidance_delta_l2.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/guidance_delta_l2_plots"

python analysis/plot_latent_step_lengths.py \
    --input_root "$INPUT_ROOT" \
    --output_dir "$OUTPUT_ROOT/latent_step_lengths"

echo "Saved best/worst saved-latent top10 analysis to: $OUTPUT_ROOT"
