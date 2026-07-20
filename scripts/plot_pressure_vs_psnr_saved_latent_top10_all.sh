#!/bin/bash

# Plot P_t_sum/R_t_sum vs PSNR after saved-latent prompt-pressure analysis has
# completed for seed-sensitive and best/worst top10 prompts.
#
# Usage:
#   bash scripts/plot_pressure_vs_psnr_saved_latent_top10_all.sh
#
# Environment overrides:
#   PROJECT_DIR=$PWD
#   OUTPUT_DIR=results/prompt_pressure_saved_latent_top10_all_analysis/pressure_vs_psnr

set -euo pipefail

#PROJECT_DIR=${PROJECT_DIR:-$PWD}
PROJECT_DIR=${PROJECT_DIR:-$PWD}

OUTPUT_DIR=${OUTPUT_DIR:-results/prompt_pressure_saved_latent_top10_all_analysis/pressure_vs_psnr}

python analysis/plot_seed_sensitive_pressure_vs_psnr.py \
    --metrics_csv \
        results/prompt_pressure_seed_sensitive_top10_analysis/distribution_metrics/per_seed_distribution_metrics.csv \
        results/prompt_pressure_best_worst_top10_saved_latents_analysis/distribution_metrics/per_seed_distribution_metrics.csv \
    --label all \
    --output_dir "$OUTPUT_DIR"

echo "Saved combined saved-latent pressure-vs-PSNR plots to: $OUTPUT_DIR"
