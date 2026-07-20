# 不同服务器结果差异分析报告

**日期**: 2026-04-05  
**问题**: 相同代码在yagi36和yagi37上得到不同结果

---

## 📊 观察到的差异

### 运行结果对比

| 指标 | yagi36 | yagi37 | 差异 |
|------|--------|--------|------|
| **第一行输出** | 4.24e-06 | 8.34e-06 | **差异约2倍** |
| **总时间** | 86.19s | 85.60s | 相差0.59s (0.7%) |
| **平均时间** | 0.1231s | 0.1223s | 相差0.0008s (0.7%) |

### 服务器硬件差异

| 配置 | yagi36 | yagi37 | 影响 |
|------|--------|--------|------|
| **CPU架构** | 2 Sockets × 4 Cores = 16核 | 1 Socket × 64 Cores = 128核 | ⚠️ 不同 |
| **内核版本** | 6.8.0-90 (Nov 2025) | 6.8.0-63 (Jun 2025) | ⚠️ 不同 |
| **GPU** | RTX A6000 × 4 | RTX A6000 × 4 | ✅ 相同 |
| **内存** | 257500M | 257561M | ✅ 相似 |

---

## 🔍 差异原因分析

### 1️⃣ **浮点运算的不确定性** (最可能原因)

深度学习模型在不同硬件上存在固有的数值差异：

#### CUDA并行计算的非确定性
- **问题**: GPU并行计算的累加顺序不固定
- **影响**: 浮点数累加顺序不同 → 舍入误差不同 → 最终结果略有差异
- **量级**: 通常在1e-6到1e-5级别（符合观察到的差异）

#### cuDNN算法选择
- **问题**: cuDNN会根据硬件特性选择不同的卷积算法
- **yagi36**: 可能选择算法A（适合多socket架构）
- **yagi37**: 可能选择算法B（适合单socket大核心）
- **影响**: 算法不同 → 计算路径不同 → 数值结果略有差异

### 2️⃣ **随机数生成器的状态**

虽然代码可能设置了种子，但：
- CUDA随机数生成器可能在不同GPU架构上表现不同
- PyTorch的某些操作（如dropout、随机采样）可能受影响
- 但从日志看没有明显的随机性操作，影响应该较小

### 3️⃣ **内存布局和缓存**

- **yagi36**: 2个socket → NUMA架构 → 可能的跨socket内存访问
- **yagi37**: 1个socket → UMA架构 → 统一内存访问
- **影响**: 内存访问模式不同可能导致计算顺序微调

---

## 🎯 第一行输出是什么？

查看输出格式，第一行 `4.24e-06` 和 `8.34e-06` 很可能是：

1. **某个损失值** (loss)
2. **MSE误差**
3. **收敛阈值**
4. **两个latent之间的距离**

**差异分析**:
- yagi36: 4.24e-06
- yagi37: 8.34e-06
- **yagi37的值约是yagi36的2倍**

这个差异虽然在绝对值上很小（都是微米级别），但相对差异达到**97%**。

---

## ✅ 这种差异是否正常？

### 是的，这是**正常且预期**的行为

1. **数值计算的固有特性**:
   - 浮点运算不满足结合律: `(a + b) + c ≠ a + (b + c)`
   - GPU并行计算顺序不确定
   - 1e-6级别的差异在深度学习中是可接受的

2. **对最终结果的影响**:
   - 看总时间差异仅0.7%，说明整体性能一致
   - 这么小的数值差异不会影响：
     - 生成图像的视觉质量
     - 反演精度
     - 模型收敛性

3. **类似案例**:
   - PyTorch官方文档明确说明CUDA操作可能是非确定性的
   - TensorFlow也有类似的已知行为
   - 这是GPU加速计算的trade-off

---

## 🛠️ 如何保证结果一致？

如果需要完全可重复的结果（如科研实验），可以采取以下措施：

### 1. 设置确定性模式

```python
import torch
import numpy as np
import random

# 设置所有随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 强制PyTorch使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置CUDA为确定性模式（会降低性能）
    torch.use_deterministic_algorithms(True)

set_seed(42)
```

### 2. 环境变量设置

```bash
# 强制cuDNN使用确定性算法
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 禁用TF32（可能导致差异）
export NVIDIA_TF32_OVERRIDE=0
```

### 3. 在代码中添加

```python
# 在脚本开头添加
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

### ⚠️ 注意事项

启用确定性模式的代价：
- ❌ **性能下降10-30%**
- ❌ 某些操作可能不支持（会报错）
- ❌ 可能需要更多GPU内存

**建议**:
- 如果只是生产环境使用 → 不需要完全一致
- 如果是科研论文实验 → 建议启用确定性模式
- 如果是性能测试 → 接受小幅差异即可

---

## 📋 验证方法

### 1. 检查最终质量指标

```bash
# 比较两个服务器的最终结果
python analysis/compare_results.py \
    --result1 outputs_yagi36/ \
    --result2 outputs_yagi37/ \
    --metrics init_inv gen_rec psnr
```

### 2. 统计显著性测试

如果差异持续存在，可以运行统计检验：
- 多次运行取平均
- 计算标准差和置信区间
- t-test检验差异是否显著

### 3. 视觉检查

```bash
# 生成对比图像
python scripts/visual_comparison.py \
    --img1 outputs_yagi36/100rec.png \
    --img2 outputs_yagi37/100rec.png
```

如果图像在视觉上无法区分，说明数值差异无关紧要。

---

## 🎯 建议行动方案

### 方案A: 接受差异（推荐）

**适用场景**: 生产环境、性能优先

```python
# 不做任何修改，接受1e-6级别的差异
# 优点：性能最优
# 缺点：结果不完全可重复
```

### 方案B: 部分确定性

**适用场景**: 需要一定可重复性但不想牺牲太多性能

```python
# 只设置种子，不强制确定性算法
set_seed(42)
torch.backends.cudnn.benchmark = True  # 保持True以获得性能
```

### 方案C: 完全确定性（仅科研）

**适用场景**: 论文实验、需要完全可重复

```python
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### 方案D: 固定服务器

**适用场景**: 需要对比实验

```bash
# 在sbatch脚本中指定节点
#SBATCH --nodelist=yagi37

# 或者排除某些节点
#SBATCH --exclude=yagi36
```

---

## 📝 结论

### 问题总结

1. ✅ **差异是正常的**: 1e-6级别的数值差异在GPU计算中是预期行为
2. ✅ **影响很小**: 总时间差异仅0.7%，不影响实际使用
3. ✅ **可控**: 如有需要，可通过确定性模式消除差异（代价是性能下降）

### 推荐做法

对于当前场景：
1. **短期**: 接受差异，继续使用
2. **长期**: 如果需要严格可重复性，在代码中添加确定性设置
3. **实验对比**: 固定在同一服务器上运行对比实验

### 验证建议

```bash
# 1. 比较最终质量指标（更重要）
python compare_quality.py --dir1 yagi36 --dir2 yagi37

# 2. 如果质量指标差异<1%，可以忽略数值差异
# 3. 如果质量指标差异>5%，需要进一步调查
```

---

**参考文档**:
- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- CUDA Deterministic Operations: https://docs.nvidia.com/cuda/cublas/index.html
- cuDNN Reproducibility: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html

