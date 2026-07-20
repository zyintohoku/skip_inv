#!/usr/bin/env python3
"""
Visualize ablation study results as a scatter plot.
X-axis: Init↔Inv -log(MSE) (inversion accuracy)
Y-axis: Gen↔Rec -log(MSE) (reconstruction quality)
Bubble size: Average time (larger = slower)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_scatter_plot(data, output_path):
    """Create scatter plot visualization."""
    # Extract method names and metrics
    methods = []
    init_inv = []
    gen_rec = []
    times = []
    colors = {
        'AFPI-0.3': '#FFD93D',  # 黄色 - 最快
        'AFPI-0.5': '#FF6B6B',  # 红色 - 快速
        'AFPI-0.7': '#4ECDC4',  # 青绿 - 平衡
        'AFPI-0.9': '#45B7D1',  # 蓝色 - 高质量
        'FPI-default': '#FFA07A', # 浅红 - 基准
        'AIDI-GS7': '#9B59B6',  # 紫色 - AIDI
    }

    for method_name, metrics in data.items():
        if method_name == 'summary':
            continue
        methods.append(method_name)
        init_inv.append(metrics['init_vs_inv_nlm'])
        gen_rec.append(metrics['gen_vs_rec_nlm'])
        times.append(metrics['avg_time_s'])

    # Normalize times for bubble sizes (scale to 300-1500)
    times_array = np.array(times)
    min_time, max_time = times_array.min(), times_array.max()
    sizes = 300 + (times_array - min_time) / (max_time - min_time) * 1200

    # Create figure with high DPI for better quality
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

    # Plot scatter points
    for i, method in enumerate(methods):
        color = colors.get(method, '#95E1D3')
        ax.scatter(init_inv[i], gen_rec[i], s=sizes[i], alpha=0.6,
                  color=color, edgecolors='black', linewidth=2)

        # Add method name as text annotation
        ax.annotate(method, (init_inv[i], gen_rec[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Customize axes
    ax.set_xlabel('Init↔Inv -log(MSE)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gen↔Rec -log(MSE)', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Inversion Methods Comparison',
                fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Create legend with uniform marker size
    handles = []
    labels = []
    for i, method in enumerate(methods):
        color = colors.get(method, '#95E1D3')
        handles.append(plt.scatter([], [], s=100, c=color, alpha=0.6,
                                  edgecolors='black', linewidth=1.5))
        labels.append(f"{method}\n({times[i]:.1f}s)")

    ax.legend(handles, labels, loc='upper right', fontsize=9,
             title='Methods', title_fontsize=10, framealpha=0.95,
             scatterpoints=1)

    # Set axis limits with some padding
    ax.set_xlim(init_inv[0] - 0.2, max(init_inv) + 0.3)
    ax.set_ylim(min(gen_rec) - 0.3, max(gen_rec) + 0.5)

    # Improve layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved to: {output_path}")

    return fig, ax

def create_comparison_table(data):
    """Print detailed comparison table."""
    print("\n" + "="*100)
    print("ABLATION STUDY - DETAILED COMPARISON")
    print("="*100 + "\n")

    print(f"{'Method':<15} {'Init↔Inv':<12} {'Gen↔Rec':<12} {'Avg Time':<12} {'Quality':<10} {'Speed':<10}")
    print("-" * 100)

    for method_name, metrics in sorted(data.items()):
        if method_name == 'summary':
            continue
        print(f"{method_name:<15} "
              f"{metrics['init_vs_inv_nlm']:<12.4f} "
              f"{metrics['gen_vs_rec_nlm']:<12.4f} "
              f"{metrics['avg_time_s']:<12.2f}s "
              f"#{metrics['init_vs_inv_rank']+metrics['gen_vs_rec_rank']//2:<10} "
              f"#{metrics['speed_rank']:<10}")

    print("\n" + "="*100)

def analyze_positions(data):
    """Analyze which quadrant each method falls into."""
    print("\n" + "="*100)
    print("QUADRANT ANALYSIS")
    print("="*100 + "\n")

    # Calculate median values
    init_inv_values = [m['init_vs_inv_nlm'] for k, m in data.items() if k != 'summary']
    gen_rec_values = [m['gen_vs_rec_nlm'] for k, m in data.items() if k != 'summary']

    median_init_inv = np.median(init_inv_values)
    median_gen_rec = np.median(gen_rec_values)

    print(f"Median Init↔Inv: {median_init_inv:.4f}")
    print(f"Median Gen↔Rec: {median_gen_rec:.4f}\n")

    print("Quadrant Distribution:")
    print(f"  High Quality [Upper Right]: Better inversion + Better reconstruction")
    print(f"  Fast & Good [Upper Left]:   Worse inversion + Better reconstruction")
    print(f"  Good Inv [Lower Right]:     Better inversion + Worse reconstruction")
    print(f"  Poor [Lower Left]:          Worse inversion + Worse reconstruction\n")

    for method_name, metrics in sorted(data.items()):
        if method_name == 'summary':
            continue

        init_inv = metrics['init_vs_inv_nlm']
        gen_rec = metrics['gen_vs_rec_nlm']
        time = metrics['avg_time_s']

        # Determine quadrant
        if init_inv >= median_init_inv and gen_rec >= median_gen_rec:
            quadrant = "Upper Right (Best Quality)"
        elif init_inv < median_init_inv and gen_rec >= median_gen_rec:
            quadrant = "Upper Left (Fast & Good)"
        elif init_inv >= median_init_inv and gen_rec < median_gen_rec:
            quadrant = "Lower Right (Good Inversion)"
        else:
            quadrant = "Lower Left (Poor)"

        print(f"{method_name:<15} → {quadrant:<30} (Time: {time:.2f}s)")

    print("\n" + "="*100)

def main():
    json_path = '/home/yzeng/remote/skip_inv/results/ablation_study/analysis/ablation_results.json'
    output_path = '/home/yzeng/remote/skip_inv/results/ablation_study/analysis/ablation_comparison.png'

    # Load data
    data = load_results(json_path)

    # Create visualization
    fig, ax = create_scatter_plot(data, output_path)

    # Print analysis
    create_comparison_table(data)
    analyze_positions(data)

    print(f"\n📊 Visualization saved to: {output_path}")
    print("\nVisualization Guide:")
    print("  • X-axis: Higher = Better inversion accuracy")
    print("  • Y-axis: Higher = Better reconstruction quality")
    print("  • Bubble size: Larger = Slower (more inference time)")
    print("  • Color & Legend: Method names with average time per sample")

if __name__ == '__main__':
    main()
