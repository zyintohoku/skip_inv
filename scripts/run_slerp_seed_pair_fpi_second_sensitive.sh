#!/bin/bash

# Submit SLERP FPI jobs for the second most seed-sensitive prompt.
#
# Source:
#   results/fpi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv rank 2
#
# sample_id=651
# prompt="the village in the game, the witcher 3"
# best seeds: 3,1
# worst seeds: 6,4
#
# Usage:
#   bash scripts/run_slerp_seed_pair_fpi_second_sensitive.sh yagi35 yagi38 yagi39
#
# Environment overrides still work, for example:
#   NUM_INTERPOLATION_POINTS=21 bash scripts/run_slerp_seed_pair_fpi_second_sensitive.sh yagi35

set -euo pipefail

export SAMPLE_ID=${SAMPLE_ID:-651}
export PROMPT=${PROMPT:-"the village in the game, the witcher 3"}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/slerp_fpi_gs7_seed_sensitive_sample0651}
export PAIR_SPECS=${PAIR_SPECS:-best1_worst1:3-6,best1_best2:3-1,worst1_worst2:6-4}

exec bash scripts/run_slerp_seed_pair_fpi.sh "$@"
