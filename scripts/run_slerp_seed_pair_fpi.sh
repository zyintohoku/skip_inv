#!/bin/bash

# Submit the seed-pair SLERP generation -> inversion -> reconstruction experiment.
#
# Defaults are taken from results/fpi_gs7_seed_psnr:
#   most sensitive sample: 443
#   best seeds: 6,8
#   worst seeds: 4,7
#   pairs: best1-worst1, best1-best2, worst1-worst2
#
# Usage:
#   bash scripts/run_slerp_seed_pair_fpi.sh
#   bash scripts/run_slerp_seed_pair_fpi.sh yagi35
#   bash scripts/run_slerp_seed_pair_fpi.sh yagi35 yagi38 yagi39
#
# Environment overrides:
#   SAMPLE_ID=443
#   PROMPT="a fox is walking in the snow"
#   ALPHAS=0,0.25,0.5,0.75,1
#   NUM_INTERPOLATION_POINTS=11
#   OUTPUT_DIR=outputs/slerp_fpi_gs7_seed_sensitive_sample0443
#   PAIR_SPECS=best1_worst1:6-4,best1_best2:6-8,worst1_worst2:4-7
#   METHOD=fpi
#   GUIDANCE_SCALE=7
#   NUM_DDIM_STEPS=50
#   TRACE_DTYPE=float16
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:2,yagi38:1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
SAMPLE_ID=${SAMPLE_ID:-}
PROMPT=${PROMPT:-}
ALPHAS=${ALPHAS:-}
NUM_INTERPOLATION_POINTS=${NUM_INTERPOLATION_POINTS:-11}
OUTPUT_DIR=${OUTPUT_DIR:-}
PAIR_SPECS=${PAIR_SPECS:-best1_worst1:6-4,best1_best2:6-8,worst1_worst2:4-7}
METHOD=${METHOD:-fpi}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}
TRACE_DTYPE=${TRACE_DTYPE:-float16}
SENSITIVE_CSV=${SENSITIVE_CSV:-results/fpi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv}
PSNR_DETAIL_CSV=${PSNR_DETAIL_CSV:-results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv}
MAPPING_FILE=${MAPPING_FILE:-PIE_bench/mapping_file.json}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

slurm_load_node_slots NODE_ARRAY "$@"

IFS=',' read -ra PAIR_ARRAY <<< "$PAIR_SPECS"
if [ ${#PAIR_ARRAY[@]} -eq 0 ]; then
    echo "Error: no pair specs parsed from PAIR_SPECS=$PAIR_SPECS"
    exit 1
fi

base_output=${OUTPUT_DIR:-outputs/slerp_fpi_gs7_seed_sensitive_sample0443}

echo "Submitting ${#PAIR_ARRAY[@]} pair job(s)."
echo "Nodes: ${NODE_ARRAY[*]}"
echo "Base output: $base_output"
echo ""

for pair_idx in "${!PAIR_ARRAY[@]}"; do
    spec=${PAIR_ARRAY[$pair_idx]}
    pair_label=${spec%%:*}
    pair_value=${spec#*:}
    if [ "$pair_label" = "$pair_value" ] || [ -z "$pair_label" ] || [ -z "$pair_value" ]; then
        echo "Error: pair spec must look like label:seedA-seedB, got '$spec'"
        exit 1
    fi

    selected_node=$(slurm_node_for_index "$pair_idx" "${NODE_ARRAY[@]}")
    partition=$(slurm_partition "$selected_node")

    job_name="slerp_fpi_${pair_label}"
    if [ -n "$SAMPLE_ID" ]; then
        job_name="slerp_fpi_s${SAMPLE_ID}_${pair_label}"
    fi
    pair_output="${base_output}/${pair_label}"

    cmd=(python run_slerp_seed_pair_fpi.py
        --method "$METHOD"
        --guidance_scale "$GUIDANCE_SCALE"
        --num_of_ddim_steps "$NUM_DDIM_STEPS"
        --delta_threshold "$DELTA_THRESHOLD"
        --loss_divergence_threshold "$LOSS_DIVERGENCE_THRESHOLD"
        --num_interpolation_points "$NUM_INTERPOLATION_POINTS"
        --trace_dtype "$TRACE_DTYPE"
        --sensitive_csv "$SENSITIVE_CSV"
        --psnr_detail_csv "$PSNR_DETAIL_CSV"
        --mapping_file "$MAPPING_FILE"
        --model_name "$MODEL_NAME"
        --pairs "$pair_value"
        --output "$pair_output")

    if [ -n "$SAMPLE_ID" ]; then
        cmd+=(--sample_id "$SAMPLE_ID")
    fi
    if [ -n "$PROMPT" ]; then
        cmd+=(--prompt "$PROMPT")
    fi
    if [ -n "$ALPHAS" ]; then
        cmd+=(--alphas "$ALPHAS")
    fi

    printf -v quoted_cmd '%q ' "${cmd[@]}"

    echo "Submitting $job_name -> $selected_node (partition=$partition, pair=$pair_value)"
    echo "Output: $pair_output"

    sbatch --job-name="$job_name" \
           --chdir="$PROJECT_DIR" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem="$MEM" \
           --cpus-per-task="$CPUS_PER_TASK" \
           --output="$LOG_DIR/${job_name}.out" \
           --error="$LOG_DIR/${job_name}.err" \
           --wrap="bash -lc 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $quoted_cmd'"
done

echo ""
echo "Submitted ${#PAIR_ARRAY[@]} job(s). Check status with: squeue -u \$(whoami)"
