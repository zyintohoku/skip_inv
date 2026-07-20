#!/bin/bash
# 高级版本：收集yagi节点信息并生成对比表格

OUTPUT_FILE="${1:-node_info_$(date +%Y%m%d_%H%M%S).txt}"
CSV_FILE="${OUTPUT_FILE%.txt}.csv"
NODES=(29 33 34 35 36 37 38 39 40 41 45)

echo "================================================================================
SLURM Node Information Collection
Date: $(date)
Nodes: yagi${NODES[@]}
================================================================================
" | tee "$OUTPUT_FILE"

# 创建CSV表头
echo "Node,State,CPUs,Memory(MB),GPUs,GPU_Type,Sockets,CoresPerSocket,OS_Version,Kernel" > "$CSV_FILE"

# 收集详细信息
for node_num in "${NODES[@]}"; do
    node_name="yagi${node_num}"
    echo -e "\n################################################################################" | tee -a "$OUTPUT_FILE"
    echo "# Node: ${node_name}" | tee -a "$OUTPUT_FILE"
    echo "################################################################################" | tee -a "$OUTPUT_FILE"
    
    # 获取节点信息
    node_info=$(scontrol show node "${node_name}" 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "$node_info" >> "$OUTPUT_FILE"
        echo "  ✓ ${node_name} collected"
        
        # 提取关键信息到CSV
        state=$(echo "$node_info" | grep -oP 'State=\K[^ ]+' | head -1)
        cpus=$(echo "$node_info" | grep -oP 'CPUTot=\K[0-9]+')
        memory=$(echo "$node_info" | grep -oP 'RealMemory=\K[0-9]+')
        gres=$(echo "$node_info" | grep -oP 'Gres=\K[^ ]+')
        gpu_count=$(echo "$gres" | grep -oP 'gpu:[^:]+:\K[0-9]+' || echo "0")
        gpu_type=$(echo "$gres" | grep -oP 'gpu:\K[^:]+' || echo "N/A")
        sockets=$(echo "$node_info" | grep -oP 'Sockets=\K[0-9]+')
        cores_per_socket=$(echo "$node_info" | grep -oP 'CoresPerSocket=\K[0-9]+')
        os_info=$(echo "$node_info" | grep -oP 'OS=\K[^ ]+')
        kernel=$(echo "$os_info" | grep -oP '^Linux [^ ]+' || echo "N/A")
        
        # 写入CSV
        echo "${node_name},${state},${cpus},${memory},${gpu_count},${gpu_type},${sockets},${cores_per_socket},${os_info},${kernel}" >> "$CSV_FILE"
    else
        echo "$node_info" >> "$OUTPUT_FILE"
        echo "  ✗ ${node_name} failed"
        echo "${node_name},ERROR,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" >> "$CSV_FILE"
    fi
done

# 生成摘要
echo -e "\n\n================================================================================
SUMMARY TABLE
================================================================================\n" | tee -a "$OUTPUT_FILE"

# 使用column命令格式化CSV为表格（如果可用）
if command -v column &> /dev/null; then
    echo "Node-wise comparison:" | tee -a "$OUTPUT_FILE"
    column -t -s ',' "$CSV_FILE" | tee -a "$OUTPUT_FILE"
else
    cat "$CSV_FILE" | tee -a "$OUTPUT_FILE"
fi

echo -e "\n================================================================================
Files created:
  - Detailed info: $OUTPUT_FILE
  - CSV table:     $CSV_FILE
================================================================================

Quick commands:
  # View full info
  cat $OUTPUT_FILE
  
  # View table only
  cat $CSV_FILE | column -t -s ','
  
  # Search for specific info
  grep 'CPUTot' $OUTPUT_FILE
  grep 'Gres' $OUTPUT_FILE
  grep 'State' $OUTPUT_FILE
  
  # Count by state
  grep -oP 'State=\K[^ ]+' $OUTPUT_FILE | sort | uniq -c
  
  # List GPU types
  grep -oP 'gpu:\K[^:]+' $OUTPUT_FILE | sort | uniq -c
================================================================================
"
