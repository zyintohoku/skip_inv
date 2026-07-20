#!/usr/bin/env python3
"""
Plot per-sample distribution of skip_inv results.
X-axis: Init↔Inv -log(MSE) per sample
Y-axis: Gen↔Rec -log(MSE) per sample
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Colors for different delta_threshold values
COLORS_DT = {
    '5e-9': '#3498DB',  # 蓝色
    '1e-9': '#E74C3C',  # 红色
}

# Markers for different force_converge values
MARKERS_FC = {
    10: 'o',
    20: 's',
    30: '^',
    40: 'D',
}


def compute_per_sample_metrics(result_dir):
    """Load latents and compute per-sample metrics."""
    init_latents = torch.load(os.path.join(result_dir, 'init_latents.pt'), map_location='cpu')
    inv_latents = torch.load(os.path.join(result_dir, 'inv_latents.pt'), map_location='cpu')
    gen_latents = torch.load(os.path.join(result_dir, 'gen_latents.pt'), map_location='cpu')
    rec_latents = torch.load(os.path.join(result_dir, 'rec_latents.pt'), map_location='cpu')
    
    init_inv_nlm = []
    gen_rec_nlm = []
    
    for init, inv, gen, rec in zip(init_latents, inv_latents, gen_latents, rec_latents):
        init_inv_mse = F.mse_loss(init, inv).item()
        gen_rec_mse = F.mse_loss(gen, rec).item()
        init_inv_nlm.append(-np.log(init_inv_mse))
        gen_rec_nlm.append(-np.log(gen_rec_mse))
    
    return np.array(init_inv_nlm), np.array(gen_rec_nlm)


def create_distribution_plot(init_inv, gen_rec, dt, fc, output_path):
    """Create scatter plot of per-sample distribution."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    color = COLORS_DT.get(dt, '#9B59B6')
    
    # Plot scatter points
    ax.scatter(init_inv, gen_rec, s=20, alpha=0.5, c=color, edgecolors='none')
    
    # Add mean point
    mean_x, mean_y = np.mean(init_inv), np.mean(gen_rec)
    ax.scatter(mean_x, mean_y, s=200, c='red', marker='*', edgecolors='black', 
               linewidth=1.5, label=f'Mean ({mean_x:.2f}, {mean_y:.2f})', zorder=5)
    
    # Customize axes
    ax.set_xlabel('Init↔Inv -log(MSE)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Gen↔Rec -log(MSE)', fontsize=13, fontweight='bold')
    ax.set_title(f'skip_inv dt={dt} fc={fc} Per-Sample Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    
    # Add statistics text
    stats_text = (f'N = {len(init_inv)}\n'
                  f'Init↔Inv: {mean_x:.2f} ± {np.std(init_inv):.2f}\n'
                  f'Gen↔Rec: {mean_y:.2f} ± {np.std(gen_rec):.2f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figure saved to: {output_path}")


def main():
    print("="*60)
    print("skip_inv Per-Sample Distribution")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Parameters to check
    delta_thresholds = ['5e-9', '1e-9']
    force_converge_steps = [10, 20, 30, 40]
    
    for dt in delta_thresholds:
        for fc in force_converge_steps:
            # Convert dt format for directory name (5e-9 -> 5e9)
            dt_short = dt.replace('e-9', 'e9').replace('5e9', 'dt5e9').replace('1e9', 'dt1e9')
            dir_name = f'skip_inv_{dt_short}_fc{fc}'
            result_dir = os.path.join(OUTPUTS_DIR, dir_name)
            
            if not os.path.exists(result_dir):
                print(f"Warning: {result_dir} not found")
                continue
            
            # Check if latent files exist
            if not os.path.exists(os.path.join(result_dir, 'init_latents.pt')):
                print(f"Warning: {result_dir}/init_latents.pt not found")
                continue
            
            print(f"\nProcessing {dir_name}...")
            init_inv, gen_rec = compute_per_sample_metrics(result_dir)
            
            print(f"  Samples: {len(init_inv)}")
            print(f"  Init↔Inv: {np.mean(init_inv):.4f} ± {np.std(init_inv):.4f}")
            print(f"  Gen↔Rec: {np.mean(gen_rec):.4f} ± {np.std(gen_rec):.4f}")
            
            output_path = os.path.join(RESULTS_DIR, f"{dir_name}_distribution.png")
            create_distribution_plot(init_inv, gen_rec, dt, fc, output_path)


if __name__ == '__main__':
    main()
