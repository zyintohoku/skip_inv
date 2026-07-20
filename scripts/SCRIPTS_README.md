# 脚本说明文档

## 主要脚本

### run.py
- **用途**: 主要的Python运行脚本
- **用法**: `python run.py [options]`
- **说明**: 支持多种实验方法和参数配置

### run.sh
- **用途**: Shell启动脚本
- **用法**: `bash run.sh`

### skip_inv.py
- **用途**: Skip失效反演相关的主要实现脚本
- **用法**: `python skip_inv.py`

---

## AIDI 实验脚本

针对不同GPU配置和节点的实验。

| 脚本 | 节点 | 配置 | 用途 |
|------|------|------|------|
| `run_aidi_gs1.sh` | 自动 | 标准配置 | AIDI GS1基础实验 |
| `run_aidi_gs1_yagi29.sh` | yagi29 | GS1配置 | AIDI GS1（指定yagi29） |
| `run_aidi_gs3.sh` | 自动 | 标准配置 | AIDI GS3基础实验 |
| `run_aidi_gs3_yagi33.sh` | yagi33 | GS3配置 | AIDI GS3（指定yagi33） |
| `run_aidi_gs5.sh` | 自动 | 标准配置 | AIDI GS5基础实验 |
| `run_aidi_gs5_yagi35.sh` | yagi35 | GS5配置 | AIDI GS5（指定yagi35） |
| `run_aidi_gs7.sh` | 自动 | 标准配置 | AIDI GS7基础实验 |
| `run_aidi_gs7_yagi36.sh` | yagi36 | GS7配置 | AIDI GS7（指定yagi36） |

**提交方式**: `sbatch <script_name>`

---

## Ablation Study 脚本

### FPI Ablation Study
| 脚本 | 节点 | 用途 |
|------|------|------|
| `run_ablation_fpi_default.sh` | yagi37 | FPI默认配置的消融研究 |

**提交方式**: `sbatch run_ablation_fpi_default.sh`

### AFPI Ablation Study（不同阈值）
| 脚本 | 阈值 | 节点 | 输出目录 |
|------|------|------|---------|
| `run_ablation_afpi_0.3.sh` | 0.3 | yagi37 | `results/ablation_study/afpi/threshold_0.3` |
| `run_ablation_afpi_0.5.sh` | 0.5 | yagi37 | `results/ablation_study/afpi/threshold_0.5` |
| `run_ablation_afpi_0.7.sh` | 0.7 | yagi37 | `results/ablation_study/afpi/threshold_0.7` |
| `run_ablation_afpi_0.9.sh` | 0.9 | yagi37 | `results/ablation_study/afpi/threshold_0.9` |

**说明**: 测试AFPI方法在不同loss_divergence_threshold（损失发散阈值）下的性能

**提交方式**: `sbatch <script_name>`

---

## 测试脚本

用于本地快速测试，**不提交到SLURM队列**，直接在当前节点运行。

| 脚本 | 用途 | 测试内容 |
|------|------|---------|
| `test_fpi.sh` | FPI方法测试 | 测试FPI默认配置 |
| `test_afpi.sh` | AFPI方法测试 | 测试AFPI的四个不同阈值（0.3, 0.5, 0.7, 0.9） |
| `test_ablation.sh` | 整体消融研究测试 | 测试所有消融研究配置 |

**用法**: `bash <script_name>`

**说明**: 用于验证配置和快速调试，所有输出到`results/ablation_study/`目录

---

## 快速命令参考

### 提交AIDI实验（推荐）
```bash
sbatch run_aidi_gs1.sh
sbatch run_aidi_gs3.sh
sbatch run_aidi_gs5.sh
sbatch run_aidi_gs7.sh
```

### 提交AFPI消融研究（推荐）
```bash
sbatch run_ablation_afpi_0.3.sh
sbatch run_ablation_afpi_0.5.sh
sbatch run_ablation_afpi_0.7.sh
sbatch run_ablation_afpi_0.9.sh
```

### 查看任务状态
```bash
squeue -u <your_username>
```

### 查看错误日志
```bash
cat log/ablation_afpi_0.3.err
```

---

## 目录结构说明

- `log/` - 脚本执行的错误日志
- `results/` - 实验结果输出目录
- `outputs/` - 额外的输出文件
- `PIE_bench/` - 基准数据集
- `utils/` - 工具函数库
