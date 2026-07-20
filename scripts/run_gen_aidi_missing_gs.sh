#!/bin/bash

# Submit generation jobs for missing AIDI GS combinations.
# Target combos:
#   - aidi_gs1 / gs1, gs3, gs5
#   - aidi_gs3 / gs3, gs5
#   - aidi_gs5 / gs5
#
# Usage:
#   bash scripts/run_gen_aidi_missing_gs.sh
#   bash scripts/run_gen_aidi_missing_gs.sh yagi36
#   bash scripts/run_gen_aidi_missing_gs.sh yagi36 yagi37 yagi40

PROJECT_DIR=${PROJECT_DIR:-$PWD}

JOBS=(
  "aidi_gs1:1"
  "aidi_gs1:3"
  "aidi_gs1:5"
  "aidi_gs3:3"
  "aidi_gs3:5"
  "aidi_gs5:5"
)

if [ $# -eq 1 ]; then
  SINGLE_NODE=$1
  echo "Using single node for all jobs: $SINGLE_NODE"
  for i in "${!JOBS[@]}"; do
    NODE_ARRAY[$i]=$SINGLE_NODE
  done
elif [ $# -ge ${#JOBS[@]} ]; then
  NODE_ARRAY=("$@")
  echo "Using specified nodes:"
  for i in "${!JOBS[@]}"; do
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

for i in "${!JOBS[@]}"; do
  pair="${JOBS[$i]}"
  method="${pair%%:*}"
  gs="${pair##*:}"
  node_idx=$((i % ${#NODE_ARRAY[@]}))
  selected_node="${NODE_ARRAY[$node_idx]}"
  partition=$(sinfo -N -h -o "%P" -n "$selected_node" | head -1)

  name="gen_${method}_gs${gs}"
  echo "  ├─ Submitting $name → $selected_node (method=$method, gs=$gs)"

  sbatch --job-name="$name" \
         --nodelist="$selected_node" \
         --partition="$partition" \
         --gres=gpu:1 \
         --mem=40G \
         --cpus-per-task=4 \
         --output="log/${name}.out" \
         --error="log/${name}.err" \
         --wrap="bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate afpi && python gen.py --base_dir outputs/reconstruction --methods ${method} --guidance_scales ${gs}'"
done

echo ""
echo "✨ All ${#JOBS[@]} missing GS generation jobs submitted!"
echo "Check status: squeue -u \$(whoami)"
