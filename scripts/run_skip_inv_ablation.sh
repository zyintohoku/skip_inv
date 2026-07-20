#!/bin/bash

# Submit skip_inv ablation: delta_threshold × force_converge_before_step
# Usage: 
#   bash scripts/run_skip_inv_ablation.sh              # Auto-select nodes (one per job)
#   bash scripts/run_skip_inv_ablation.sh yagi36       # All jobs on one node
#   bash scripts/run_skip_inv_ablation.sh yagi33 yagi35 yagi36 yagi37 ...  # Specify nodes (need 8)

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Parameters to test
DELTA_THRESHOLDS=(5e-9 1e-9)
FORCE_CONVERGE_STEPS=(10 20 30 40)

# Generate all configurations: delta_threshold × force_converge_before_step
CONFIGS=()
for dt in "${DELTA_THRESHOLDS[@]}"; do
    for fc in "${FORCE_CONVERGE_STEPS[@]}"; do
        # Create short name: dt5e9_fc10, dt1e9_fc20, etc.
        dt_short=$(echo "$dt" | sed 's/e-9/e9/' | sed 's/5e9/dt5e9/' | sed 's/1e9/dt1e9/')
        name="${dt_short}_fc${fc}"
        CONFIGS+=("${dt}:${fc}:${name}")
    done
done

echo "Ablation: delta_threshold × force_converge_before_step"
echo "Total configurations: ${#CONFIGS[@]}"
echo ""

if [ $# -eq 1 ]; then
    # Single node for all jobs
    SINGLE_NODE=$1
    echo "Using single node for all jobs: $SINGLE_NODE"
    for i in "${!CONFIGS[@]}"; do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
elif [ $# -ge 1 ]; then
    # Use specified nodes in round-robin fashion
    SPECIFIED_NODES=("$@")
    echo "Using specified nodes (round-robin):"
    for i in "${!CONFIGS[@]}"; do
        NODE_ARRAY[$i]=${SPECIFIED_NODES[$((i % $#))]}
    done
    printf "  %s\n" "${SPECIFIED_NODES[@]}"
else
    # Auto-select nodes
    echo "Searching for available nodes..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(29|3[3-9]|4[0-1])" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -lt ${#CONFIGS[@]} ]; then
        echo "❌ Need ${#CONFIGS[@]} nodes but only found ${#NODE_ARRAY[@]}!"
        echo "Using first available node for all jobs..."
        SINGLE_NODE="${NODE_ARRAY[0]}"
        for i in "${!CONFIGS[@]}"; do
            NODE_ARRAY[$i]=$SINGLE_NODE
        done
    else
        echo "✅ Found ${#NODE_ARRAY[@]} available nodes, using first ${#CONFIGS[@]}:"
        for i in "${!CONFIGS[@]}"; do
            echo "  ${NODE_ARRAY[$i]}"
        done
    fi
fi
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    selected_node="${NODE_ARRAY[$i]}"
    IFS=':' read -r delta_threshold force_converge name <<< "$config"
    
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    
    echo "  ├─ Submitting skip_inv $name → $selected_node (dt=$delta_threshold, fc=$force_converge)"

    sbatch --job-name=skip_${name} \
           --nodelist=$selected_node \
           --partition=$partition \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --output=log/skip_${name}.out \
           --error=log/skip_${name}.err \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python skip_inv.py --delta_threshold $delta_threshold --force_converge_before_step $force_converge --output outputs/skip_inv_${name}'"
done

echo ""
echo "✨ All ${#CONFIGS[@]} skip_inv jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
