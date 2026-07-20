#!/bin/bash

# Test AIDI on multiple nodes
# Usage: 
#   bash scripts/run_test_aidi.sh              # Auto-select nodes
#   bash scripts/run_test_aidi.sh yagi35       # Single node
#   bash scripts/run_test_aidi.sh yagi35 yagi38 yagi39  # Specific nodes

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Default test nodes
TEST_NODES=(35 38 39 41)

if [ $# -eq 1 ]; then
    # Single node for all jobs
    SINGLE_NODE=$1
    echo "Using single node: $SINGLE_NODE"
    for i in "${!TEST_NODES[@]}"; do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
elif [ $# -ge ${#TEST_NODES[@]} ]; then
    # Use specified nodes
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for node in "${NODE_ARRAY[@]}"; do
        echo "  $node"
    done
else
    # Use default nodes
    echo "Using default nodes:"
    for node_num in "${TEST_NODES[@]}"; do
        NODE_ARRAY+=("yagi${node_num}")
        echo "  yagi${node_num}"
    done
fi
echo ""

for node in "${NODE_ARRAY[@]}"; do
    name="test_aidi_${node}"
    
    # Get partition for this node
    partition=$(sinfo -N -h -o "%P" -n "$node" 2>/dev/null | head -1)
    
    if [ -z "$partition" ]; then
        echo "  ⚠️  Cannot determine partition for $node, using default"
        partition="48-4"
    fi
    
    echo "  ├─ Submitting $name → $node (partition=$partition)"

    sbatch --job-name=$name \
           --nodelist=$node \
           --partition=$partition \
           --gres=gpu:1 \
           --mem=48G \
           --cpus-per-task=4 \
           --output=log/${name}.out \
           --error=log/${name}.err \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python test_aidi.py --method aidi'"
done

echo ""
echo "✨ All ${#NODE_ARRAY[@]} jobs submitted!"
echo "Check status: squeue -u \$(whoami)"
echo "View output: tail -f log/test_aidi_yagi*.out"
