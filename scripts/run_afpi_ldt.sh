#!/bin/bash

# Test AFPI with different loss_divergence_threshold values
# Usage: 
#   bash scripts/run_afpi_ldt.sh              # Auto-select nodes (one per job)
#   bash scripts/run_afpi_ldt.sh yagi36       # All jobs on one node
#   bash scripts/run_afpi_ldt.sh yagi37 yagi40 yagi45 yagi29  # Specify nodes

PROJECT_DIR=${PROJECT_DIR:-$PWD}

LDT_VALUES=(0.3 0.5 0.7 0.9)

if [ $# -eq 1 ]; then
    # Single node for all jobs
    SINGLE_NODE=$1
    echo "Using single node for all jobs: $SINGLE_NODE"
    for i in "${!LDT_VALUES[@]}"; do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
elif [ $# -ge ${#LDT_VALUES[@]} ]; then
    # Use specified nodes
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!LDT_VALUES[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
else
    # Auto-select nodes
    echo "Searching for available nodes..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(37|40|45)" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ] || [ -z "${NODE_ARRAY[0]}" ]; then
        echo "❌ No available nodes found!"
        sinfo -N -h -o "%N %P %t" | grep -E "yagi(37|40|45)"
        exit 1
    fi

    echo "✅ Found ${#NODE_ARRAY[@]} available nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  ${NODE_ARRAY[$i]}"
    done
fi
echo ""

for i in "${!LDT_VALUES[@]}"; do
    ldt="${LDT_VALUES[$i]}"
    # Cycle through available nodes
    node_idx=$((i % ${#NODE_ARRAY[@]}))
    selected_node="${NODE_ARRAY[$node_idx]}"
    name="afpi_ldt${ldt//.}"
    
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    
    echo "  ├─ Submitting $name → $selected_node (ldt=$ldt)"

    sbatch --job-name=$name \
           --nodelist=$selected_node \
           --partition=$partition \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --output=log/${name}.out \
           --error=log/${name}.err \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python run.py --method afpi --loss_divergence_threshold $ldt --output outputs/${name}'"
done

echo ""
echo "✨ All ${#LDT_VALUES[@]} jobs submitted!"
echo "Check status: squeue -u \$(whoami)"
