#!/bin/bash

# Submit AFPI ablation study and FPI comparison experiments
# Automatically finds available nodes and submits jobs

PROJECT_DIR=${PROJECT_DIR:-$PWD}

# Define target nodes
TARGET_NODES="yagi29,yagi33,yagi34,yagi35,yagi36,yagi37,yagi38,yagi39,yagi40,yagi41"

# Define experiments: (method, params_description, param_arg)
EXPERIMENTS=(
    "afpi 0.9 --method afpi --loss_divergence_threshold 0.9"
    "afpi 0.7 --method afpi --loss_divergence_threshold 0.7"
    "afpi 0.5 --method afpi --loss_divergence_threshold 0.5"
    "fpi default --method fpi"
)

# Check if node is specified as argument
if [ -z "$1" ]; then
    # Get list of idle nodes in TARGET_NODES range
    echo "Searching for available nodes in range yagi29,33-41..."
    AVAILABLE_NODES=$(sinfo -N -h -o "%N %t" | grep -E "yagi(29|3[3-9]|4[0-1])" | awk '$2 ~ /idle|mix/ {print $1}' | sort -V)

    mapfile -t NODE_ARRAY <<< "$AVAILABLE_NODES"

    if [ ${#NODE_ARRAY[@]} -eq 0 ]; then
        echo "❌ No available nodes found! Current status:"
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

echo "📌 All ${#EXPERIMENTS[@]} jobs will be submitted to: $selected_node (partition: $partition)"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    read -r method desc params <<< "${EXPERIMENTS[$i]}"

    echo "  ├─ Submitting ${method} (${desc})"

    # Create temporary SLURM script for this job
    SCRIPT_FILE="run_ablation_${method}_${desc}.sh"

    # Determine output directory based on method
    if [[ "$method" == "afpi" ]]; then
        output_dir="results/ablation_study/afpi/threshold_${desc}"
    else
        output_dir="results/ablation_study/fpi"
    fi

    cat > "$SCRIPT_FILE" << SCRIPT_EOF
#!/bin/bash
#SBATCH --job-name=ablation_${method}_${desc}
#SBATCH --nodelist=$selected_node
#SBATCH --partition=$partition
###########RESOURCES###########
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
###############################
#SBATCH --output=log/ablation_${method}_${desc}.out
#SBATCH --error=log/ablation_${method}_${desc}.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate afpi
##############################
python run.py ${params} --output ${output_dir}
SCRIPT_EOF

    chmod +x "$SCRIPT_FILE"
    sbatch "$SCRIPT_FILE"
done

echo ""
echo "✨ All ${#EXPERIMENTS[@]} jobs submitted!"
echo "Check status with: squeue -u \$(whoami)"
