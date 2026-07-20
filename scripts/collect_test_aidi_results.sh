#!/bin/bash

# Collect AIDI test results from multiple nodes

PROJECT_DIR=${PROJECT_DIR:-$PWD}

NODES=(35 38 39 41)

echo "================================================================================"
echo "AIDI Test Results"
echo "================================================================================"
echo ""

for node_num in "${NODES[@]}"; do
    node="yagi${node_num}"
    out_file="log/test_aidi_${node}.out"
    
    echo "Node: $node"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    if [ ! -f "$out_file" ]; then
        echo "  ❌ Output file not found: $out_file"
        echo ""
        continue
    fi
    
    # Extract results
    if [ -s "$out_file" ]; then
        echo "  Results:"
        cat "$out_file"
        echo ""
    else
        echo "  ⚠️  File is empty or job not completed"
        echo ""
    fi
done

echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
echo "Value 1        Total Time    Avg Time      Node"
echo "────────────────────────────────────────────────────────────────────────────────"

for node_num in "${NODES[@]}"; do
    node="yagi${node_num}"
    out_file="log/test_aidi_${node}.out"
    
    if [ -f "$out_file" ] && [ -s "$out_file" ]; then
        value1=$(sed -n '1p' "$out_file" | grep -oP '[0-9]+\.[0-9]+e-[0-9]+' || echo "N/A")
        total_time=$(grep "total_time:" "$out_file" | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        avg_time=$(grep "avg_time:" "$out_file" | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        
        printf "%-14s %-13s %-13s %s\n" "$value1" "$total_time" "$avg_time" "$node"
    else
        printf "%-14s %-13s %-13s %s (not ready)\n" "N/A" "N/A" "N/A" "$node"
    fi
done

echo ""
echo "================================================================================"
