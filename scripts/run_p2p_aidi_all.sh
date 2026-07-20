#!/bin/bash

# Submit P2P editing for AIDI GS1/3/5/7 with configurable nodes.
# Usage:
#   bash scripts/run_p2p_aidi_all.sh                        # Auto-select nodes
#   bash scripts/run_p2p_aidi_all.sh yagi36                # Single node for all jobs
#   bash scripts/run_p2p_aidi_all.sh yagi36 yagi37 yagi40  # Multiple nodes (cycled)

PROJECT_DIR=${PROJECT_DIR:-$PWD}

LATENT_GS_VALUES=(1 3 5 7)
GUIDANCE_VALUES=(1 3 5 7)

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

job_idx=0
for latent_gs in "${LATENT_GS_VALUES[@]}"; do
    latent_dir="outputs/reconstruction/aidi_gs${latent_gs}"
    if [ ! -d "$latent_dir" ]; then
        echo "⚠ Skip missing latent dir: $latent_dir"
        continue
    fi

    for guidance_scale in "${GUIDANCE_VALUES[@]}"; do
        if [ "$guidance_scale" -lt "$latent_gs" ]; then
            continue
        fi

        node_idx=$((job_idx % ${#NODE_ARRAY[@]}))
        selected_node="${NODE_ARRAY[$node_idx]}"
        partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

        output_dir="outputs/editing/aidi_gs${latent_gs}/gs${guidance_scale}"
        mkdir -p "$output_dir"

        job_name="p2p_aidi_lgs${latent_gs}_gs${guidance_scale}"
        echo "  ├─ Submitting ${job_name} → ${selected_node} (partition: ${partition})"
        echo "      latent_dir=${latent_dir}"
        echo "      output_dir=${output_dir}"

        sbatch --job-name="$job_name" \
               --nodelist="$selected_node" \
               --partition="$partition" \
               --gres=gpu:1 \
               --mem=30G \
               --cpus-per-task=4 \
               --output="log/${job_name}.out" \
               --error="log/${job_name}.err" \
               --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && cd p2p && python p2p.py --latent_dir ../${latent_dir} --output_dir ../${output_dir} --guidance_scale ${guidance_scale}'"

        job_idx=$((job_idx + 1))
    done
done

echo ""
echo "✨ All eligible P2P jobs submitted!"
echo "Rules: for latent aidi_gsX, run guidance_scale in [1,3,5,7] where gs >= X."
echo "Check status: squeue -u \$(whoami)"
