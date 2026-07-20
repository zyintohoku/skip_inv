#!/bin/bash
# 快速查看分析结果

echo "=========================================="
echo "Ablation Study 分析结果"
echo "=========================================="
echo ""

if [ -f "results/ablation_study/analysis/ablation_report.md" ]; then
    echo "📊 详细报告："
    cat results/ablation_study/analysis/ablation_report.md | head -60
    echo ""
    echo "[完整报告: results/ablation_study/analysis/ablation_report.md]"
else
    echo "❌ 报告文件不存在"
fi

echo ""
echo "=========================================="
echo "快速查询"
echo "=========================================="
echo ""
echo "✓ 查看完整 Markdown 报告:"
echo "  cat results/ablation_study/analysis/ablation_report.md"
echo ""
echo "✓ 查看 JSON 格式数据:"
echo "  cat results/ablation_study/analysis/ablation_results.json"
echo ""
echo "✓ 查看 CSV 表格:"
echo "  cat results/ablation_study/analysis/ablation_results.csv"
echo ""
echo "✓ 重新生成分析报告:"
echo "  cd analysis && python3 generate_report.py"
echo ""
