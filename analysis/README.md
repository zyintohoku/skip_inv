# Analysis Scripts

本目录包含用于分析实验结果的 Python 脚本。

## 文件说明

### Ablation Study (AFPI/FPI 方法对比)

| 文件 | 功能 | 输出 |
|------|------|------|
| `generate_report.py` | 计算 latent 空间 metrics (Init↔Inv, Gen↔Rec)，生成报告 | `results/ablation_study/` |
| `plot_ablation.py` | 生成 ablation 散点图 (x=反演精度, y=重建质量, size=时间) | `ablation_comparison.png` |
| `plot_threshold_comparison.py` | 生成阈值曲线图，比较不同方法在各阈值下的表现 | `threshold_*.png` |

### AIDI Analysis

| 文件 | 功能 | 输出 |
|------|------|------|
| `analyze_aidi.py` | 分析 AIDI 不同 guidance scale (GS1-7) 的 latent metrics | 终端输出 |
| `plot_aidi.py` | AIDI GS1-7 散点图 (x=反演精度, y=重建质量) | `results/aidi_gs_comparison.png` |
| `plot_aidi_distribution.py` | AIDI 各 GS 的 per-sample 分布图 | `results/aidi_gs{1,3,5,7}_distribution.png` |

### P2P Editing Evaluation

| 文件 | 功能 | 输出 |
|------|------|------|
| `evaluate_p2p.py` | 计算 P2P 编辑结果的 CLIP/PSNR/SSIM/LPIPS metrics | `results/p2p_*.json`, `results/p2p_*.md` |
| `plot_p2p_results.py` | 从缓存数据生成 P2P 图表（无需重新计算） | `results/clip_threshold_*.png` |

## 使用方法

```bash
# Ablation study
python analysis/generate_report.py
python analysis/plot_ablation.py
python analysis/plot_threshold_comparison.py

# AIDI analysis
python analysis/analyze_aidi.py
python analysis/plot_aidi.py
python analysis/plot_aidi_distribution.py

# P2P evaluation
python analysis/evaluate_p2p.py      # 完整计算（耗时）
python analysis/plot_p2p_results.py  # 仅生成图表（快速）
```

## 更新日志

- 2026-03-31: 初始版本，整理目录结构
