#!/bin/bash

# Submit AIDI experiments with fixed guidance_scale=7 and seeds 1..10
# Automatically finds available nodes from TARGET_NODES and submits jobs

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Define target nodes
TARGET_NODES="yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"
SEEDS=($(seq 1 10))
GUIDANCE_SCALE=7

# Parse node arguments
if [ $# -eq 0 ]; then
    # Get list of idle nodes from TARGET_NODES
    echo "Searching for available nodes in TARGET_NODES: $TARGET_NODES"
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | awk -v nodes="$TARGET_NODES" '
        BEGIN {
            split(nodes, arr, ",");
            for (i in arr) allow[arr[i]] = 1;
        }
        $2 ~ /idle|mix/ && ($1 in allow) { print $1 }
    ' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ]; then
        echo "❌ No available nodes found! Current status:"
        sinfo -N -h -o "%N %P %t" | awk -v nodes="$TARGET_NODES" '
            BEGIN {
                split(nodes, arr, ",");
                for (i in arr) allow[arr[i]] = 1;
            }
            ($1 in allow) { print }
        '
        exit 1
    fi

    echo "✅ Found ${#NODE_ARRAY[@]} available nodes:"
    printf '%s\n' "${NODE_ARRAY[@]}"
    echo ""
else
    # Use specified nodes (single or multiple)
    NODE_ARRAY=("$@")
    echo "Using specified nodes:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  Job $i: ${NODE_ARRAY[$i]}"
    done
    echo ""
fi

echo "📌 Total jobs: ${#SEEDS[@]}"
echo "📌 Nodes in rotation: ${#NODE_ARRAY[@]}"
echo "📌 guidance_scale fixed at: $GUIDANCE_SCALE"
echo ""

job_idx=0
for seed in "${SEEDS[@]}"; do
    NODE_IDX=$((job_idx % ${#NODE_ARRAY[@]}))
    selected_node=${NODE_ARRAY[$NODE_IDX]}
    partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)
    if [ -z "$partition" ]; then
        echo "  ⚠️  Cannot determine partition for $selected_node, using default 48-4"
        partition="48-4"
    fi
    echo "  ├─ Submitting seed=$seed → $selected_node (partition=$partition)"

    # Create temporary SLURM script for this job
    SCRIPT_FILE="run_aidi_gs${GUIDANCE_SCALE}_seed${seed}.sh"

    cat > "$SCRIPT_FILE" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=aidi_gs${GUIDANCE_SCALE}_s${seed}
#SBATCH --nodelist=$selected_node
#SBATCH --partition=$partition
###########RESOURCES###########
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
###############################
#SBATCH --output=log/aidi_gs${GUIDANCE_SCALE}_s${seed}.out
#SBATCH --error=log/aidi_gs${GUIDANCE_SCALE}_s${seed}.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate afpi
##############################
python run.py --method aidi --guidance_scale ${GUIDANCE_SCALE} --seed ${seed} --output outputs/aidi_gs${GUIDANCE_SCALE}_seed${seed}
SCRIPT_EOF

    chmod +x "$SCRIPT_FILE"
    sbatch "$SCRIPT_FILE"
    job_idx=$((job_idx + 1))
done

echo ""
echo "✨ All ${#SEEDS[@]} jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
