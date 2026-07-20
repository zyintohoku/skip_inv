"""
示例：如何在skip_inv.py中添加可重复性设置

将以下代码添加到skip_inv.py的开头（在import语句之后）
"""

# ============================================================================
# 可重复性设置示例
# ============================================================================

# 方法1: 使用自定义工具函数（推荐）
from utils.reproducibility import set_reproducibility

# 在main函数开始时调用
def main(...):
    # 基础模式：设置种子但不强制确定性（性能最优）
    config = set_reproducibility(seed=42, strict=False)
    
    # 或者使用严格模式（完全可重复但性能下降10-30%）
    # config = set_reproducibility(seed=42, strict=True)
    
    # ... 其余代码 ...


# ============================================================================
# 方法2: 直接在代码中设置（如果不想使用工具函数）
# ============================================================================

import torch
import numpy as np
import random
import os

# 在main函数开始时添加
def main(...):
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 如果需要完全确定性（会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # 如果只需要基本的可重复性（推荐，性能更好）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # ... 其余代码 ...


# ============================================================================
# 方法3: 通过命令行参数控制
# ============================================================================

def main(
        output_dir='output',
        guidance_scale=7,
        num_of_ddim_steps=50,
        delta_threshold=5e-12,
        loss_divergence_threshold=1.0,
        reset_gs=False,
        seed=None,  # 新增参数
        deterministic=False,  # 新增参数
        **kwargs
):
    # 如果提供了种子，设置可重复性
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print(f"✅ Deterministic mode enabled with seed={seed}")
        else:
            torch.backends.cudnn.benchmark = True
            print(f"✅ Seed set to {seed} (non-deterministic mode for better performance)")
    
    # ... 其余代码 ...


# 对应的argparse设置：
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... 其他参数 ...
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable fully deterministic mode (slower but reproducible)')
    args = parser.parse_args()


# ============================================================================
# 在SLURM脚本中使用
# ============================================================================

"""
# 方法A: 固定服务器（推荐，最简单）
#SBATCH --nodelist=yagi37

# 方法B: 使用确定性模式
python skip_inv.py --seed 42 --deterministic

# 方法C: 基本种子设置（性能更好）
python skip_inv.py --seed 42
"""


# ============================================================================
# 性能对比
# ============================================================================

"""
模式                    | 性能     | 可重复性       | 推荐场景
-----------------------|----------|----------------|------------------
无设置                  | 100%     | ❌ 不保证      | 性能测试
基础种子（方法2默认）   | 95-100%  | ⚠️  大致一致   | 生产环境
严格确定性              | 70-90%   | ✅ 完全一致    | 科研论文
固定服务器              | 100%     | ✅ 完全一致    | 对比实验（推荐）
"""

print(__doc__)
