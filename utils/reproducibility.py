"""
工具函数：设置PyTorch确定性模式以保证跨服务器结果一致
"""
import torch
import numpy as np
import random
import os

def set_reproducibility(seed=42, strict=False):
    """
    设置可重复性模式
    
    Args:
        seed: 随机种子
        strict: 是否使用严格模式（会降低性能但保证完全一致）
    
    Returns:
        配置信息字典
    """
    config = {
        'seed': seed,
        'strict_mode': strict,
        'performance_impact': 'High' if strict else 'Low'
    }
    
    # 1. 设置Python、NumPy、PyTorch随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if strict:
        # 严格模式：完全确定性，但性能下降
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 强制使用确定性算法
        try:
            torch.use_deterministic_algorithms(True)
            config['deterministic_algorithms'] = True
        except AttributeError:
            # PyTorch < 1.8 不支持
            config['deterministic_algorithms'] = False
        
        # 设置环境变量
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        config['cudnn_deterministic'] = True
        config['cudnn_benchmark'] = False
        
        print("✅ Strict reproducibility mode enabled")
        print("⚠️  Performance may decrease by 10-30%")
        
    else:
        # 非严格模式：设置种子但允许性能优化
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        config['cudnn_deterministic'] = False
        config['cudnn_benchmark'] = True
        
        print("✅ Basic reproducibility mode enabled")
        print("ℹ️  Results may have minor numerical differences (~1e-6) across different hardware")
    
    config['torch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['cudnn_version'] = torch.backends.cudnn.version()
        config['gpu_name'] = torch.cuda.get_device_name(0)
    
    return config


def print_reproducibility_config(config):
    """打印当前配置"""
    print("\n" + "="*60)
    print("Reproducibility Configuration")
    print("="*60)
    for key, value in config.items():
        print(f"  {key:<30}: {value}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test reproducibility settings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode')
    args = parser.parse_args()
    
    config = set_reproducibility(seed=args.seed, strict=args.strict)
    print_reproducibility_config(config)
    
    # 简单测试
    print("Running simple test...")
    x = torch.randn(10, 10).cuda() if torch.cuda.is_available() else torch.randn(10, 10)
    y = torch.randn(10, 10).cuda() if torch.cuda.is_available() else torch.randn(10, 10)
    z = torch.mm(x, y)
    print(f"Test result sum: {z.sum().item():.10f}")
    print("\n✅ Reproducibility settings applied successfully!")
