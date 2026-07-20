# 项目目录结构

```
skip_inv/
├── scripts/                      # 项目运行脚本
│   ├── run.py                   # 主程序入口
│   ├── run.sh                    # 运行脚本
│   ├── run_ablation_*.sh        # Ablation 研究脚本（AFPI 0.3-0.9）
│   ├── run_aidi_*.sh            # AIDI 对比脚本
│   └── test_*.sh                # 测试脚本
│
├── analysis/                     # 数据分析脚本
│   ├── ablation_analysis.py     # Ablation 结果分析（计算 MSE）
│   ├── simple_analysis.py       # 简化版分析脚本（无 pandas 依赖）
│   ├── generate_report.py       # 生成详细报告（Markdown + JSON）
│   └── analyze_ablation.py      # 早期版本分析脚本
│
├── results/ablation_study/       # 实验结果
│   ├── afpi/                     # AFPI 方法结果
│   │   ├── threshold_0.3/
│   │   ├── threshold_0.5/        # 700 个样本
│   │   ├── threshold_0.7/        # 700 个样本
│   │   └── threshold_0.9/        # 700 个样本
│   ├── fpi/                      # FPI 方法结果（700 个样本）
│   └── analysis/                 # 分析结果汇总
│       ├── ablation_report.md    # Markdown 报告（详细分析）
│       ├── ablation_results.csv  # CSV 格式表格
│       └── ablation_results.json # JSON 结构化数据
│
├── outputs/                      # 生成的图像和临时输出
├── log/                          # 运行日志
│   ├── ablation_afpi_0.*.out    # 运行时间日志
│   └── *.err                     # 错误日志
│
├── utils/                        # 实用工具模块
├── PIE_bench/                    # 数据集
├── skip_inv.py                   # 核心模块
└── SCRIPTS_README.md             # 脚本说明文档
```

## 快速使用

### 查看分析结果
```bash
# Markdown 格式（推荐阅读）
cat results/ablation_study/analysis/ablation_report.md

# JSON 格式（便于程序处理）
cat results/ablation_study/analysis/ablation_results.json

# CSV 格式（用于表格）
cat results/ablation_study/analysis/ablation_results.csv
```

### 重新运行分析
```bash
cd analysis

# 完整分析报告
python3 generate_report.py

# 简单分析（仅表格）
python3 simple_analysis.py

# 详细分析（pandas 版本，需要依赖）
python3 ablation_analysis.py
```

### 运行实验
```bash
cd scripts

# 运行所有 Ablation 实验
bash run_ablation_afpi_0.5.sh
bash run_ablation_afpi_0.7.sh
bash run_ablation_afpi_0.9.sh
bash run_ablation_fpi_default.sh
```

## 关键文件说明

### 分析脚本对比

| 脚本 | 功能 | 依赖 | 输出格式 |
|------|------|------|---------|
| `generate_report.py` | 完整报告生成 | torch, numpy | Markdown + JSON |
| `simple_analysis.py` | 快速表格分析 | torch, numpy | 表格输出 |
| `ablation_analysis.py` | 详细 pandas 分析 | torch, numpy, pandas | Markdown + 表格 |

### 结果文件详解

- **ablation_report.md**: 包含详细的排名、权衡分析和建议
- **ablation_results.json**: 完整的结构化数据，包含排名、评分等
- **ablation_results.csv**: 简化的表格，便于 Excel 或其他工具处理

## 最新分析结果摘要

### 核心对比（700 个样本）

| 方法 | Init↔Inv | Gen↔Rec | Avg Time | 质量排名 |
|------|----------:|-------:|-------:|-------:|
| AFPI-0.9 | 9.0259 | 10.1269 | 64.21s | 🥇 最佳 |
| AFPI-0.7 | 8.6479 | 9.3085 | 50.44s | 🥈 平衡 |
| FPI-def | 8.9579 | 7.6101 | 61.25s | 🥉 基准 |
| AFPI-0.5 | 8.1453 | 8.1808 | 43.85s | 🚀 最快 |

### 建议使用场景
- **最佳质量**: AFPI-0.9
- **最快速度**: AFPI-0.5
- **平衡方案**: AFPI-0.7 ⭐ 推荐
