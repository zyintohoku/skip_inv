#!/bin/bash

# Submit P2P editing using init_latents.pt as upper bound
# Usage: bash scripts/run_p2p_upper_bound.sh [node]
# Example: bash scripts/run_p2p_upper_bound.sh yagi36

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Use any latent_dir that has init_latents.pt (they're all the same)
LATENT_DIR="outputs/reconstruction/aidi_gs7"
OUTPUT_DIR="outputs/editing/upper_bound"

# Check if node is specified as argument
if [ -z "$1" ]; then
    echo "Searching for available nodes..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(29|3[3-9]|4[0-1])" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ]; then
        echo "❌ No available nodes found!"
        sinfo -N -h -o "%N %P %t" | grep -E "yagi(29|3[3-9]|4[0-1])"
        exit 1
    fi

    echo "✅ Found ${#NODE_ARRAY[@]} available nodes:"
    printf '%s\n' "${NODE_ARRAY[@]}"
    echo ""

    selected_node=${NODE_ARRAY[0]}
else
    selected_node=$1
    echo "Using specified node: $selected_node"
fi

partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

echo "📌 Submitting P2P upper bound experiment to: $selected_node (partition: $partition)"
echo "   Using init_latents.pt from: $LATENT_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo ""

sbatch --job-name=p2p_upper \
       --nodelist=$selected_node \
       --partition=$partition \
       --gres=gpu:1 \
       --mem=30G \
       --cpus-per-task=4 \
       --output=log/p2p_upper_bound.out \
       --error=log/p2p_upper_bound.err \
       --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && cd p2p && python p2p.py --latent_dir ../${LATENT_DIR} --output_dir ../${OUTPUT_DIR} --use_init'"

echo ""
echo "✨ Upper bound experiment submitted!"
echo "Check status with: squeue -u \$(whoami)"
