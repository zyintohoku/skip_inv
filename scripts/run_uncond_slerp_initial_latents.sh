#!/bin/bash

# Submit unconditional generation jobs from SLERP-interpolated initial latents.
#
# It submits two independent jobs by default:
#   sample 443: pairs best1_worst1:6-4,best1_best2:6-8,worst1_worst2:4-7
#   sample 651: pairs best1_worst1:3-6,best1_best2:3-1,worst1_worst2:6-4
#
# Usage:
#   bash scripts/run_uncond_slerp_initial_latents.sh
#   bash scripts/run_uncond_slerp_initial_latents.sh yagi35 yagi38
#
# Environment overrides:
#   ALPHAS=0,0.25,0.5,0.75,1
#   NUM_INTERPOLATION_POINTS=11
#   SAVE_TRACE_TENSORS=1
#   CONDA_ENV=afpi
#   PROJECT_DIR=$PWD
#   NODE_TASKS=yagi35:2,yagi38:1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/lib/slurm_common.sh"
slurm_prepare_log_dir

TARGET_NODES=${TARGET_NODES:-yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41}
ALPHAS=${ALPHAS:-}
NUM_INTERPOLATION_POINTS=${NUM_INTERPOLATION_POINTS:-11}
NUM_DDIM_STEPS=${NUM_DDIM_STEPS:-50}
TRACE_DTYPE=${TRACE_DTYPE:-float16}
MODEL_NAME=${MODEL_NAME:-CompVis/stable-diffusion-v1-4}
INIT_LATENT_ROOT_TEMPLATE=${INIT_LATENT_ROOT_TEMPLATE:-outputs/aidi_gs7_seed{seed}}
CONDA_ENV=${CONDA_ENV:-afpi}
MEM=${MEM:-48G}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SAVE_TRACE_TENSORS=${SAVE_TRACE_TENSORS:-0}

TASK_SPECS=(
    "0443|443|best1_worst1:6-4,best1_best2:6-8,worst1_worst2:4-7|outputs/uncond_slerp_initial_latents_sample0443"
    "0651|651|best1_worst1:3-6,best1_best2:3-1,worst1_worst2:6-4|outputs/uncond_slerp_initial_latents_sample0651"
)

slurm_load_node_slots NODE_ARRAY "$@"

echo "Submitting ${#TASK_SPECS[@]} unconditional SLERP job(s)."
echo "Nodes: ${NODE_ARRAY[*]}"
echo ""

for task_idx in "${!TASK_SPECS[@]}"; do
    IFS='|' read -r sample_tag sample_id pair_specs output_dir <<< "${TASK_SPECS[$task_idx]}"
    selected_node=$(slurm_node_for_index "$task_idx" "${NODE_ARRAY[@]}")
    partition=$(slurm_partition "$selected_node")

    job_name="uncond_slerp_s${sample_tag}"
    cmd=(python generate_uncond_slerp_initial_latents.py
        --sample_id "$sample_id"
        --pair_specs "$pair_specs"
        --num_interpolation_points "$NUM_INTERPOLATION_POINTS"
        --num_of_ddim_steps "$NUM_DDIM_STEPS"
        --trace_dtype "$TRACE_DTYPE"
        --model_name "$MODEL_NAME"
        --init_latent_root_template "$INIT_LATENT_ROOT_TEMPLATE"
        --output "$output_dir")

    if [ -n "$ALPHAS" ]; then
        cmd+=(--alphas "$ALPHAS")
    fi
    if [ "$SAVE_TRACE_TENSORS" = "1" ]; then
        cmd+=(--save_trace_tensors)
    fi

    printf -v quoted_cmd '%q ' "${cmd[@]}"

    echo "Submitting $job_name -> $selected_node (partition=$partition)"
    echo "Output: $output_dir"

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
echo "Submitted ${#TASK_SPECS[@]} job(s). Check status with: squeue -u \$(whoami)"
