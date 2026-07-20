# skip_pipe.py 新功能说明

## 📌 修改内容

 `utils/skip_pipe.py` 中的 `Inversion` 类添加了新参数 `force_converge_before_step`，可以控制早期步骤的收敛行为。

---

## 🎯 新增参数

### `force_converge_before_step` (可选)

**类型**: `int` 或 `None`  
**默认值**: `None`  
**说明**: 如果设置，当 `step_index < force_converge_before_step` 时，`afpi_step` 方法总是返回 `converged=True`

---

## 💡 使用方法

### 基本用法

```python
from utils.skip_pipe import Inversion

# 创建Inversion实例，设置前10步强制收敛
inversion = Inversion(
    model=pipe,
    num_ddim_steps=50,
    delta_threshold=5e-12,
    loss_divergence_threshold=0.7,
    reset_gs=True,
    force_converge_before_step=10  # ← 新
)
```

### 向后兼容

新功能说明新功能说明 `force_converge_before_step`（或设置为 `None`），代码行为完全保持原样。

---

## 🚀 应用场景

### 1. 加速早期步骤

```python
vim combine2.py 40步精细优化
inversion = Inversion(
    model=pipe,
    num_ddim_steps=50,
    force_converge_before_step=10
)
```

**预期效果**: PIE_bench README_STRUCTURE.md SCRIPTS_README.md ablation_report.md ablation_results.json analysis docs log outputs p2p results run.py script_output.txt scripts skip_inv.py test_aidi.py test_skip_inv.py utils view_results.sh 1次迭代即收敛，后期步骤正常优化

---

**修改日期**: 2026-04-04
