#!/bin/bash

# Submit AIDI jobs with guidance_scale 1,3,5, each split into 2 batches
# Total: 6 jobs
# Usage:
#   bash scripts/run_aidi_135.sh yagi36
#   bash scripts/run_aidi_135.sh yagi33 yagi34 yagi35 yagi36 yagi37 yagi38

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Guidance scales to test
GS_VALUES=(1 3 5)
BATCHES_PER_GS=2
TOTAL_SAMPLES=700
BATCH_SIZE=$((TOTAL_SAMPLES / BATCHES_PER_GS))

# Parameters
DELTA_THRESHOLD=${DELTA_THRESHOLD:-5e-12}
METHOD="aidi"

echo "="*60
echo "AIDI Batch Jobs (GS: 1, 3, 5)"
echo "="*60
echo "Each GS split into $BATCHES_PER_GS batches"
echo "Total jobs: $((${#GS_VALUES[@]} * BATCHES_PER_GS))"
echo ""

# Parse node arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: No nodes specified!"
    echo "Usage: bash scripts/run_aidi_135.sh <node1> [node2] ..."
    exit 1
elif [ $# -eq 1 ]; then
    # Single node for all jobs
    SINGLE_NODE=$1
    echo "Using single node for all jobs: $SINGLE_NODE"
    for i in {0..5}; do
        NODE_ARRAY[$i]=$SINGLE_NODE
    done
else
    # Multiple nodes
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  Job $i: ${NODE_ARRAY[$i]}"
    done
fi

echo ""

job_idx=0

# Loop through each guidance scale
for gs in "${GS_VALUES[@]}"; do
    echo "📊 Guidance Scale: $gs"
    
    # Split into batches
    for batch in $(seq 0 $((BATCHES_PER_GS - 1))); do
        # Calculate sample range
        START=$((batch * BATCH_SIZE))
        if [ $batch -eq $((BATCHES_PER_GS - 1)) ]; then
            END=$TOTAL_SAMPLES
        else
            END=$(((batch + 1) * BATCH_SIZE))
        fi
        
        # Select node
        NODE_IDX=$((job_idx % ${#NODE_ARRAY[@]}))
        SELECTED_NODE=${NODE_ARRAY[$NODE_IDX]}
        
        # Get partition
        PARTITION=$(sinfo -N -h -o "%P" -n "$SELECTED_NODE" | head -1)
        
        # Job name and output
        JOB_NAME="aidi_gs${gs}_b${batch}"
        OUTPUT_DIR="outputs/aidi_gs${gs}"
        
        echo "  ├─ Batch $batch: samples $START-$((END-1)) → $SELECTED_NODE"
        
        # Submit job
        sbatch --job-name=$JOB_NAME \
               --nodelist=$SELECTED_NODE \
               --partition=$PARTITION \
               --gres=gpu:1 \
               --mem=48G \
               --cpus-per-task=4 \
               --output=log/${JOB_NAME}.out \
               --error=log/${JOB_NAME}.err \
               --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python run_batch.py --method $METHOD --guidance_scale $gs --delta_threshold $DELTA_THRESHOLD --sample_start $START --sample_end $END --output $OUTPUT_DIR --batch_id $batch'"
        
        job_idx=$((job_idx + 1))
    done
    echo ""
done

echo "✨ All 6 jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
echo ""
echo "Results will be saved to:"
echo "  - outputs/aidi_gs1/"
echo "  - outputs/aidi_gs3/"
echo "  - outputs/aidi_gs5/"
echo ""
echo "To merge results for each GS after completion:"
echo "  python scripts/merge_batch_results.py --method aidi_gs1 --num_batches 2"
echo "  python scripts/merge_batch_results.py --method aidi_gs3 --num_batches 2"
echo "  python scripts/merge_batch_results.py --method aidi_gs5 --num_batches 2"
