#!/bin/bash

# Submit AIDI experiments with different guidance scales
# Automatically finds available nodes from TARGET_NODES and submits jobs

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Define target nodes
TARGET_NODES="yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"
GS_VALUES=(1 3 5 7)

# Check if node is specified as argument
if [ -z "$1" ]; then
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

    selected_node=${NODE_ARRAY[0]}
else
    selected_node=$1
    echo "Using specified node: $selected_node"
fi

partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

echo "📌 All ${#GS_VALUES[@]} jobs will be submitted to: $selected_node (partition: $partition)"
echo ""

for i in "${!GS_VALUES[@]}"; do
    gs=${GS_VALUES[$i]}

    echo "  ├─ Submitting guidance_scale=$gs"

    # Create temporary SLURM script for this job
    SCRIPT_FILE="run_aidi_gs${gs}.sh"

    cat > "$SCRIPT_FILE" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=aidi_gs${gs}
#SBATCH --nodelist=$selected_node
#SBATCH --partition=$partition
###########RESOURCES###########
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
###############################
#SBATCH --output=log/aidi_gs${gs}.out
#SBATCH --error=log/aidi_gs${gs}.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate afpi
##############################
python run.py --method aidi --guidance_scale $gs --output outputs/aidi_gs${gs}
SCRIPT_EOF

    chmod +x "$SCRIPT_FILE"
    sbatch "$SCRIPT_FILE"
done

echo ""
echo "✨ All ${#GS_VALUES[@]} jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
