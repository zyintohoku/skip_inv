#!/bin/bash

# Submit FPI gen-inv-rec jobs for the 6 added modifier contexts over seeds 1..10.
#
# Usage:
#   bash scripts/run_modifier_extra_context_fpi_gen_inv_rec.sh
#   bash scripts/run_modifier_extra_context_fpi_gen_inv_rec.sh yagi35 yagi38

# Run from the current repository directory. Do not cd to a fixed server path.
PROJECT_DIR=${PROJECT_DIR:-$PWD}

export PROMPT_CSV=${PROMPT_CSV:-results/fpi_gs7_seed_psnr/prompt_structure_analysis/modifier_extra_context_prompt_grid.csv}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/modifier_extra_context_fpi_gs7_seed}
export RESULTS_DIR=${RESULTS_DIR:-results/fpi_gs7_seed_psnr/modifier_prompt_grid_fpi}
export RESULT_NAME=${RESULT_NAME:-modifier_extra_context_fpi}
export JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-modextra_fpi_s}
export AGG_JOB_NAME=${AGG_JOB_NAME:-modextra_fpi_agg}

exec bash scripts/run_modifier_prompt_grid_fpi_gen_inv_rec.sh "$@"
