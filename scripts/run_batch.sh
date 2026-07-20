#!/bin/bash

# Submit batch inversion jobs split into 6 parts
# Usage:
#   bash scripts/run_batch.sh <node1> <node2> <node3> <node4> <node5> <node6>
#   bash scripts/run_batch.sh yagi33 yagi34 yagi35 yagi36 yagi37 yagi38
#
# Or specify a single node for all jobs:
#   bash scripts/run_batch.sh yagi36

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Total samples to split
TOTAL_SAMPLES=700
NUM_BATCHES=6
BATCH_SIZE=$((TOTAL_SAMPLES / NUM_BATCHES))
REMAINDER=$((TOTAL_SAMPLES % NUM_BATCHES))

# Method and parameters (can be overridden via environment variables)
METHOD=${METHOD:-"aidi"}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7}
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
LOSS_DIVERGENCE_THRESHOLD=${LOSS_DIVERGENCE_THRESHOLD:-0.9}

echo "="*60
echo "Batch Inversion Job Submission"
echo "="*60
echo "Method: $METHOD"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Total Samples: $TOTAL_SAMPLES"
echo "Number of Batches: $NUM_BATCHES"
echo "Batch Size: ~$BATCH_SIZE"
echo ""

# Parse node arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: No nodes specified!"
    echo "Usage: bash scripts/run_batch.sh <node1> <node2> ... <node6>"
    echo "   or: bash scripts/run_batch.sh <single_node>"
    exit 1
elif [ $# -eq 1 ]; then
    # Single node for all batches
    SINGLE_NODE=$1
    echo "Using single node for all batches: $SINGLE_NODE"
    for i in $(seq 0 $((NUM_BATCHES - 1))); do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
else
    # Multiple nodes
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  Batch $i: ${NODE_ARRAY[$i]}"
    done
fi

# Ensure we have enough nodes
if [ ${#NODE_ARRAY[@]} -lt $NUM_BATCHES ]; then
    echo "⚠ Warning: Only ${#NODE_ARRAY[@]} nodes specified, but $NUM_BATCHES batches needed."
    echo "Will cycle through available nodes."
fi

echo ""

# Unified output directory
OUTPUT_DIR="outputs/${METHOD}"
echo "All results will save to: $OUTPUT_DIR"
echo ""

# Submit jobs for each batch
for i in $(seq 0 $((NUM_BATCHES - 1))); do
    # Calculate sample range
    START=$((i * BATCH_SIZE))
    if [ $i -eq $((NUM_BATCHES - 1)) ]; then
        # Last batch gets remainder
        END=$TOTAL_SAMPLES
    else
        END=$(((i + 1) * BATCH_SIZE))
    fi
    
    # Select node (cycle if not enough nodes)
    NODE_IDX=$((i % ${#NODE_ARRAY[@]}))
    SELECTED_NODE=${NODE_ARRAY[$NODE_IDX]}
    
    # Get partition for selected node
    PARTITION=$(sinfo -N -h -o "%P" -n "$SELECTED_NODE" | head -1)
    
    # Job name
    JOB_NAME="${METHOD}_batch${i}"
    
    echo "📦 Batch $i: samples $START-$((END-1)) → $SELECTED_NODE"
    
    # Submit job
    sbatch --job-name=$JOB_NAME \
           --nodelist=$SELECTED_NODE \
           --partition=$PARTITION \
           --gres=gpu:1 \
           --mem=40G \
           --cpus-per-task=4 \
           --output=log/${JOB_NAME}.out \
           --error=log/${JOB_NAME}.err \
           --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python run_batch.py --method $METHOD --guidance_scale $GUIDANCE_SCALE --delta_threshold $DELTA_THRESHOLD --loss_divergence_threshold $LOSS_DIVERGENCE_THRESHOLD --sample_start $START --sample_end $END --output $OUTPUT_DIR --batch_id $i'"
done

echo ""
echo "✨ All $NUM_BATCHES batch jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
echo ""
echo "Results: $OUTPUT_DIR/"
echo "  - Images: {id}gen.png, {id}rec.png"
echo "  - Latents: init_latents_batch{i}.pt, ..."
echo ""
echo "To merge latent results after completion, run:"
echo "  python scripts/merge_batch_results.py --method $METHOD"
