#!/bin/bash
#SBATCH --job-name=test_aidi_{NODE}
#SBATCH --nodelist={NODE}
#SBATCH --output=log/test_aidi_{NODE}.out
#SBATCH --error=log/test_aidi_{NODE}.err
#SBATCH --partition=48-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00

# ============================================================================
# SLURM Job Script: Test AIDI method on specific node
# Node: {NODE}
# Command: python test_aidi.py --method aidi
# ============================================================================

echo "================================================================================"
echo "Job Information"
echo "================================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "GPUs:          $SLURM_GPUS"
echo "Start Time:    $(date)"
echo "================================================================================"
echo ""

# Print node information
echo "Node Hardware Information:"
echo "  CPU Info: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Change to working directory
PROJECT_DIR=${PROJECT_DIR:-/po1/yan/skip_inv}
cd "$PROJECT_DIR"

# Activate environment if needed
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env

# Print Python and PyTorch info
echo "Environment Information:"
echo "  Python: $(python --version 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo ""

echo "================================================================================"
echo "Running Test"
echo "================================================================================"
echo "Command: python test_aidi.py --method aidi"
echo ""

# Run the test
python test_aidi.py --method aidi

exit_code=$?

echo ""
echo "================================================================================"
echo "Job Complete"
echo "================================================================================"
echo "Exit Code:     $exit_code"
echo "End Time:      $(date)"
echo "================================================================================"

exit $exit_code
