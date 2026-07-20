#!/bin/bash
# 简洁版本：收集yagi节点信息

OUTPUT_FILE="${1:-node_info_$(date +%Y%m%d_%H%M%S).txt}"
NODES=(29 33 34 35 36 37 38 39 40 41 43 45)

echo "Collecting node information..." | tee "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

for node_num in "${NODES[@]}"; do
    node_name="yagi${node_num}"
    echo "=== ${node_name} ===" | tee -a "$OUTPUT_FILE"
    scontrol show node "${node_name}" >> "$OUTPUT_FILE" 2>&1 && echo "✓" || echo "✗"
    echo "" >> "$OUTPUT_FILE"
done

echo "Done! Output: $OUTPUT_FILE"
