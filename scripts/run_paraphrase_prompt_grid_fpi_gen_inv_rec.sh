#!/bin/bash

# Submit FPI gen-inv-rec jobs for the paraphrase prompt grid over seeds 1..10.
#
# Usage:
#   bash scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh
#   bash scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh yagi38 yagi40 yagi41
#
# Useful overrides:
#   NODE_JOB_COUNTS=2,1,1 bash scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh yagi38 yagi40 yagi41
#   SEEDS_SPEC=1-4 bash scripts/run_paraphrase_prompt_grid_fpi_gen_inv_rec.sh yagi38

PROJECT_DIR=${PROJECT_DIR:-$PWD}

export PROMPT_CSV=${PROMPT_CSV:-results/fpi_gs7_seed_psnr/paraphrase_prompt_grid/paraphrase_prompt_grid.csv}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-outputs/paraphrase_prompt_grid_fpi_gs7_seed}
export RESULTS_DIR=${RESULTS_DIR:-results/fpi_gs7_seed_psnr/paraphrase_prompt_grid}
export RESULT_NAME=${RESULT_NAME:-paraphrase_prompt_grid_fpi}
export JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-para_fpi_s}
export AGG_JOB_NAME=${AGG_JOB_NAME:-para_fpi_agg}

exec bash scripts/run_modifier_prompt_grid_fpi_gen_inv_rec.sh "$@"
