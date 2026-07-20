#!/bin/bash

# Submit P2P editing for skip_inv_dt* runs with configurable nodes.
# Usage:
#   bash scripts/run_p2p_skip_inv_all.sh                        # Auto-select nodes
#   bash scripts/run_p2p_skip_inv_all.sh yagi36                # Single node for all jobs
#   bash scripts/run_p2p_skip_inv_all.sh yagi36 yagi37 yagi40  # Multiple nodes (cycled)

PROJECT_DIR=${PROJECT_DIR:-$PWD}

if [ $# -eq 1 ]; then
    SINGLE_NODE=$1
    echo "Using single node for all jobs: $SINGLE_NODE"
    NODE_ARRAY=("$SINGLE_NODE")
elif [ $# -ge 2 ]; then
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
else
    echo "Searching for available nodes..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(29|3[3-9]|4[0-5])" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)
    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ] || [ -z "${NODE_ARRAY[0]}" ]; then
        echo "❌ No available nodes found!"
        sinfo -N -h -o "%N %P %t" | grep -E "yagi(29|3[3-9]|4[0-5])"
        exit 1
    fi

    echo "✅ Found ${#NODE_ARRAY[@]} available nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
fi

echo ""

mapfile -t LATENT_DIRS < <(find outputs/reconstruction -maxdepth 1 -type d -name 'skip_inv_dt*' | sort -V)

if [ ${#LATENT_DIRS[@]} -eq 0 ]; then
    echo "❌ No latent dirs found under outputs/reconstruction matching skip_inv_dt*"
    exit 1
fi

job_idx=0
for latent_dir in "${LATENT_DIRS[@]}"; do
    latent_name=$(basename "$latent_dir")
    output_dir="outputs/editing/${latent_name}"
    cfg_schedules_path="${latent_dir}/cfg_schedules.pt"
    mkdir -p "$output_dir"

    node_idx=$((job_idx % ${#NODE_ARRAY[@]}))
    selected_node="${NODE_ARRAY[$node_idx]}"
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

    job_name="p2p_${latent_name}"
    echo "  ├─ Submitting ${job_name} → ${selected_node} (partition: ${partition})"
    echo "      latent_dir=${latent_dir}"
    echo "      cfg_schedules=${cfg_schedules_path}"
    echo "      output_dir=${output_dir}"

    sbatch --job-name="$job_name" \
           --nodelist="$selected_node" \
           --partition="$partition" \
           --gres=gpu:1 \
           --mem=24G \
           --cpus-per-task=4 \
           --output="log/${job_name}.out" \
           --error="log/${job_name}.err" \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && cd p2p && python p2p.py --latent_dir ../${latent_dir} --output_dir ../${output_dir} --cfg_schedules_path ../${cfg_schedules_path}'"

    job_idx=$((job_idx + 1))
done

echo ""
echo "✨ All skip_inv_dt* P2P jobs submitted!"
echo "Latent source: outputs/reconstruction/skip_inv_dt*"
echo "Output target: outputs/editing/skip_inv_dt*"
echo "Check status: squeue -u \$(whoami)"
