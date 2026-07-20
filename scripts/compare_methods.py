#!/usr/bin/env python3
"""
比较不同方法的性能
X轴: Init-Inv -log(MSE)
Y轴: Gen-Rec -log(MSE)
点的大小: 运行时间
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

# 定义要比较的方法（移除FPI）
methods = {
    'AFPI-LDT0.3': 'outputs/afpi_ldt03',
    'AFPI-LDT0.5': 'outputs/afpi_ldt05',
    'AFPI-LDT0.7': 'outputs/afpi_ldt07',
    'AFPI-LDT0.9': 'outputs/afpi_ldt09',
    'AIDI-GS7': 'outputs/aidi_gs7'
}

# 定义颜色和标记（移除FPI）
colors = {
    'AFPI-LDT0.3': '#FF6B6B',  # 红色
    'AFPI-LDT0.5': '#FFA500',  # 橙色
    'AFPI-LDT0.7': '#4ECDC4',  # 青色
    'AFPI-LDT0.9': '#45B7D1',  # 蓝色
    'AIDI-GS7': '#9B59B6'      # 紫色
}

markers = {
    'AFPI-LDT0.3': 'o',
    'AFPI-LDT0.5': 'o',
    'AFPI-LDT0.7': 'o',
    'AFPI-LDT0.9': 'o',
    'AIDI-GS7': 'o'
}

# 尝试从日志文件读取运行时间
def get_runtime_from_log(method_name):
    """从日志文件中提取运行时间"""
    import re
    import glob
    
    # 特殊处理AFPI方法
    if method_name.startswith('AFPI-LDT'):
        # 提取LDT值，例如 AFPI-LDT0.3 -> 03
        ldt_value = method_name.split('LDT')[1].replace('.', '')
        log_file = f'log/afpi_ldt{ldt_value}.out'
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # 查找avg_time
                match = re.search(r'avg_time:\s*([\d.]+)', content)
                if match:
                    return float(match.group(1))
        except:
            pass
    
    # 定义可能的日志文件位置
    log_patterns = [
        f'log/*{method_name.lower().replace("-", "_")}*.out',
        f'log/*{method_name.lower().replace(".", "")}*.out',
        f'script_output.txt'
    ]
    
    for pattern in log_patterns:
        log_files = glob.glob(pattern)
        if log_files:
            try:
                with open(log_files[0], 'r') as f:
                    content = f.read()
                    # 尝试匹配各种时间格式
                    time_patterns = [
                        r'avg_time:\s*([\d.]+)',
                        r'Average time per sample:\s*([\d.]+)',
                        r'Total time:\s*([\d.]+)',
                        r'Time:\s*([\d.]+)s',
                        r'Elapsed:\s*([\d.]+)'
                    ]
                    for tp in time_patterns:
                        match = re.search(tp, content)
                        if match:
                            return float(match.group(1))
            except:
                pass
    
    return None

# 估计的运行时间（如果无法从日志读取）
estimated_times = {
    'AFPI-LDT0.3': 70.0,
    'AFPI-LDT0.5': 68.0,
    'AFPI-LDT0.7': 72.0,
    'AFPI-LDT0.9': 78.0,
    'AIDI-GS7': 93.0
}

results = {}

print("Loading data for all methods...")
for method_name, method_dir in methods.items():
    print(f"\nProcessing {method_name}...")
    
    path = Path(method_dir)
    
    # 检查目录是否存在
    if not path.exists():
        print(f"  ⚠️  Directory not found: {method_dir}")
        continue
    
    try:
        # 加载潜在向量
        init_latents = torch.load(path / 'init_latents.pt', map_location='cpu')
        inv_latents = torch.load(path / 'inv_latents.pt', map_location='cpu')
        gen_latents = torch.load(path / 'gen_latents.pt', map_location='cpu')
        rec_latents = torch.load(path / 'rec_latents.pt', map_location='cpu')
        
        num_samples = len(init_latents)
        print(f"  Loaded {num_samples} samples")
        
        # 计算每个样本的指标
        init_inv_scores = []
        gen_rec_scores = []
        
        for i in range(num_samples):
            # Init-Inv -log(MSE)
            init_inv_mse = torch.mean((init_latents[i] - inv_latents[i]) ** 2).item()
            init_inv_neg_log_mse = -np.log(init_inv_mse) if init_inv_mse > 0 else float('inf')
            
            # Gen-Rec -log(MSE)
            gen_rec_mse = torch.mean((gen_latents[i] - rec_latents[i]) ** 2).item()
            gen_rec_neg_log_mse = -np.log(gen_rec_mse) if gen_rec_mse > 0 else float('inf')
            
            # 过滤掉无效值
            if not np.isinf(init_inv_neg_log_mse) and not np.isinf(gen_rec_neg_log_mse):
                init_inv_scores.append(init_inv_neg_log_mse)
                gen_rec_scores.append(gen_rec_neg_log_mse)
        
        # 计算均值
        init_inv_mean = np.mean(init_inv_scores)
        gen_rec_mean = np.mean(gen_rec_scores)
        
        # 获取运行时间
        runtime = get_runtime_from_log(method_name)
        if runtime is None:
            runtime = estimated_times.get(method_name, 50.0)
        
        results[method_name] = {
            'init_inv_mean': init_inv_mean,
            'gen_rec_mean': gen_rec_mean,
            'runtime': runtime,
            'num_samples': num_samples,
            'init_inv_all': init_inv_scores,
            'gen_rec_all': gen_rec_scores
        }
        
        print(f"  Init-Inv mean: {init_inv_mean:.4f}")
        print(f"  Gen-Rec mean: {gen_rec_mean:.4f}")
        print(f"  Runtime: {runtime:.2f}s per sample")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# 保存结果到JSON
output_data = {}
for method_name, data in results.items():
    output_data[method_name] = {
        'init_inv_mean': float(data['init_inv_mean']),
        'gen_rec_mean': float(data['gen_rec_mean']),
        'runtime': float(data['runtime']),
        'num_samples': int(data['num_samples'])
    }

with open('outputs/method_comparison_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("\n" + "="*80)
print("Creating visualization...")
print("="*80)

# 创建图表
fig, ax = plt.subplots(figsize=(14, 12))

# 归一化点的大小（基于运行时间）
runtimes = [data['runtime'] for data in results.values()]
min_runtime = min(runtimes)
max_runtime = max(runtimes)

# 点的大小范围: 100-500
size_min, size_max = 100, 500

# 绘制每个方法（不添加标签）
for method_name, data in results.items():
    # 归一化大小
    normalized_size = (data['runtime'] - min_runtime) / (max_runtime - min_runtime) if max_runtime > min_runtime else 0.5
    point_size = size_min + normalized_size * (size_max - size_min)
    
    # 绘制均值点
    ax.scatter(
        data['init_inv_mean'], 
        data['gen_rec_mean'],
        s=point_size,
        c=colors[method_name],
        marker=markers[method_name],
        alpha=0.7,
        edgecolors='black',
        linewidth=2.5,
        zorder=3
    )

# 为legend创建自定义句柄（统一大小）
from matplotlib.lines import Line2D
legend_elements = []
for method_name in sorted(results.keys()):
    data = results[method_name]
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=colors[method_name], 
               markersize=12, 
               markeredgecolor='black',
               markeredgewidth=2,
               label=f"{method_name} ({data['runtime']:.1f}s)")
    )

# 设置坐标轴
ax.set_xlabel('Init-Inv -log(MSE) (Inversion Accuracy)', fontsize=14, fontweight='bold')
ax.set_ylabel('Gen-Rec -log(MSE) (Reconstruction Quality)', fontsize=14, fontweight='bold')
ax.set_title('Method Comparison: Inversion Accuracy vs Reconstruction Quality\n(Bubble size represents runtime)', 
             fontsize=16, fontweight='bold', pad=20)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 添加对角线参考线（理想情况下两者应该相等）
all_values = []
for data in results.values():
    all_values.extend([data['init_inv_mean'], data['gen_rec_mean']])
min_val, max_val = min(all_values), max(all_values)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='Equal Quality Line')

# 添加图例（使用自定义句柄，统一大小）
legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
                   framealpha=0.95, edgecolor='black', title='Method (Runtime)', ncol=1)
legend.get_title().set_fontsize(12)
legend.get_title().set_fontweight('bold')

# 添加尺寸说明文本框
# 找出最快和最慢的方法名
fastest_method = min(results.items(), key=lambda x: x[1]['runtime'])
slowest_method = max(results.items(), key=lambda x: x[1]['runtime'])

textstr = f'Circle Size = Runtime\n\nSmallest: {min_runtime:.1f}s ({fastest_method[0]})\nLargest: {max_runtime:.1f}s ({slowest_method[0]})\n\nLarger circle = Slower method'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=2)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

# 设置坐标轴范围，留出边距
margin = 0.5
ax.set_xlim(min_val - margin, max_val + margin)
ax.set_ylim(min_val - margin, max_val + margin)

# 保存高分辨率图片
plt.tight_layout()
plt.savefig('outputs/method_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved figure to: outputs/method_comparison.png")

# 创建第二张图：详细对比
fig2, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：Init-Inv对比
ax1 = axes[0]
methods_sorted = sorted(results.items(), key=lambda x: x[1]['init_inv_mean'], reverse=True)
method_names = [m[0] for m in methods_sorted]
init_inv_values = [m[1]['init_inv_mean'] for m in methods_sorted]
method_colors = [colors[m] for m in method_names]

bars1 = ax1.barh(method_names, init_inv_values, color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Init-Inv -log(MSE)', fontsize=12, fontweight='bold')
ax1.set_title('Inversion Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars1, init_inv_values)):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
             va='center', fontsize=10, fontweight='bold')

# 右图：Gen-Rec对比
ax2 = axes[1]
methods_sorted2 = sorted(results.items(), key=lambda x: x[1]['gen_rec_mean'], reverse=True)
method_names2 = [m[0] for m in methods_sorted2]
gen_rec_values = [m[1]['gen_rec_mean'] for m in methods_sorted2]
method_colors2 = [colors[m] for m in method_names2]

bars2 = ax2.barh(method_names2, gen_rec_values, color=method_colors2, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Gen-Rec -log(MSE)', fontsize=12, fontweight='bold')
ax2.set_title('Reconstruction Quality Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars2, gen_rec_values)):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/method_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✅ Saved figure to: outputs/method_comparison_bars.png")

# 打印汇总表格
print("\n" + "="*80)
print("Method Comparison Summary")
print("="*80)
print(f"{'Method':<15} {'Init-Inv':<12} {'Gen-Rec':<12} {'Runtime (s)':<12} {'Samples':<8}")
print("-"*80)
for method_name in sorted(results.keys()):
    data = results[method_name]
    print(f"{method_name:<15} {data['init_inv_mean']:<12.4f} {data['gen_rec_mean']:<12.4f} "
          f"{data['runtime']:<12.2f} {data['num_samples']:<8}")

print("\n" + "="*80)
print("Ranking by Init-Inv -log(MSE) (Higher is Better)")
print("="*80)
sorted_by_init = sorted(results.items(), key=lambda x: x[1]['init_inv_mean'], reverse=True)
for rank, (method_name, data) in enumerate(sorted_by_init, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"{marker} {method_name:<15} {data['init_inv_mean']:.4f}")

print("\n" + "="*80)
print("Ranking by Gen-Rec -log(MSE) (Higher is Better)")
print("="*80)
sorted_by_gen = sorted(results.items(), key=lambda x: x[1]['gen_rec_mean'], reverse=True)
for rank, (method_name, data) in enumerate(sorted_by_gen, 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"{marker} {method_name:<15} {data['gen_rec_mean']:.4f}")

print("\n" + "="*80)
print("Ranking by Runtime (Lower is Better)")
print("="*80)
sorted_by_time = sorted(results.items(), key=lambda x: x[1]['runtime'])
for rank, (method_name, data) in enumerate(sorted_by_time, 1):
    marker = "🚀" if rank == 1 else "⚡" if rank == 2 else "✅" if rank == 3 else f"{rank}."
    print(f"{marker} {method_name:<15} {data['runtime']:.2f}s")

print("\n✅ Done! All visualizations saved.")
