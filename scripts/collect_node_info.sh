#!/bin/bash
# 脚本：收集指定yagi节点的信息
# 用途：通过scontrol show node命令获取节点详细信息并保存

OUTPUT_FILE="node_info_$(date +%Y%m%d_%H%M%S).txt"

# 节点列表
NODES=(29 33 34 35 36 37 38 39 40 41 43 45)

echo "================================================================================"
echo "Collecting node information from SLURM cluster"
echo "Date: $(date)"
echo "================================================================================"
echo ""

# 创建输出文件
cat > "$OUTPUT_FILE" << EOF
================================================================================
SLURM Node Information Collection
Generated: $(date)
Nodes: yagi${NODES[@]}
================================================================================

EOF

# 遍历所有节点
for node_num in "${NODES[@]}"; do
    node_name="yagi${node_num}"
    echo "Collecting info for ${node_name}..."
    
    # 添加分隔符和节点名称
    cat >> "$OUTPUT_FILE" << EOF

################################################################################
# Node: ${node_name}
################################################################################
EOF
    
    # 获取节点信息
    scontrol show node "${node_name}" >> "$OUTPUT_FILE" 2>&1
    
    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "  ✓ ${node_name} - Success"
    else
        echo "  ✗ ${node_name} - Failed (may not exist or not accessible)"
    fi
    
    echo "" >> "$OUTPUT_FILE"
done

# 添加总结
cat >> "$OUTPUT_FILE" << EOF

================================================================================
Summary
================================================================================
Total nodes queried: ${#NODES[@]}
Collection completed: $(date)
================================================================================
EOF

echo ""
echo "================================================================================"
echo "Collection complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "================================================================================"
echo ""
echo "To view the file:"
echo "  cat $OUTPUT_FILE"
echo ""
echo "To search for specific information:"
echo "  grep 'CPUTot' $OUTPUT_FILE"
echo "  grep 'Gres' $OUTPUT_FILE"
echo "  grep 'State' $OUTPUT_FILE"
echo ""
