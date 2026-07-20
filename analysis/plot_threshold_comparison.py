#!/usr/bin/env python3
"""
Plot comparison of different methods across different thresholds.
Shows the percentage of samples with metrics above each threshold.

Init↔Inv: Uses -log(MSE) of latents
PSNR & SSIM: Uses generated (gen) and reconstructed (rec) images
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Try to import skimage for SSIM, otherwise use alternative
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except:
    HAS_SKIMAGE = False
    print("Warning: skimage not available, will compute SSIM using pytorch")

def compute_psnr(img1, img2):
    """Compute PSNR between two images (numpy arrays, values in [0, 1] or [0, 255])."""
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)

    # Normalize to [0, 1] if needed
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)

    # Normalize to [0, 1] if needed
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0

    if HAS_SKIMAGE:
        # Use skimage if available (handles multi-channel images better)
        if len(img1.shape) == 3:
            # For RGB images, compute SSIM for each channel and average
            return ssim(img1, img2, channel_axis=2, data_range=1.0)
        else:
            return ssim(img1, img2, data_range=1.0)
    else:
        # Fallback: convert to grayscale and compute
        from PIL import Image as PILImage
        img1_pil = PILImage.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = PILImage.fromarray((img2 * 255).astype(np.uint8))
        img1_gray = np.array(img1_pil.convert('L'), dtype=np.float32) / 255.0
        img2_gray = np.array(img2_pil.convert('L'), dtype=np.float32) / 255.0

        # Simple SSIM computation
        c1, c2 = 0.01, 0.03
        mean1 = np.mean(img1_gray)
        mean2 = np.mean(img2_gray)
        var1 = np.var(img1_gray)
        var2 = np.var(img2_gray)
        cov = np.mean((img1_gray - mean1) * (img2_gray - mean2))

        ssim_val = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        return ssim_val

def compute_init_inv_metrics(result_dir):
    """Load latents and compute MSE metrics."""
    try:
        init_latents = torch.load(os.path.join(result_dir, 'init_latents.pt'),
                                 map_location=torch.device('cpu'))
        inv_latents = torch.load(os.path.join(result_dir, 'inv_latents.pt'),
                                map_location=torch.device('cpu'))

        init_inv_mse_list = []

        for init, inv in zip(init_latents, inv_latents):
            init_inv_mse = F.mse_loss(init, inv).item()
            init_inv_mse_list.append(init_inv_mse)

        # Compute -log(MSE) for each sample
        init_inv_nlm = -np.log(np.array(init_inv_mse_list))
        return init_inv_nlm

    except Exception as e:
        print(f"Error processing {result_dir}: {e}")
        return None

def compute_image_metrics(result_dir):
    """Load image files and compute PSNR and SSIM metrics."""
    try:
        psnr_list = []
        ssim_list = []

        # Find all gen.png files
        gen_files = sorted([f for f in os.listdir(result_dir) if f.endswith('gen.png')])

        for gen_file in gen_files:
            idx = gen_file.replace('gen.png', '')
            rec_file = f'{idx}rec.png'

            gen_path = os.path.join(result_dir, gen_file)
            rec_path = os.path.join(result_dir, rec_file)

            if not os.path.exists(rec_path):
                continue

            # Load images
            img_gen = Image.open(gen_path).convert('RGB')
            img_rec = Image.open(rec_path).convert('RGB')

            # Compute metrics
            psnr = compute_psnr(img_gen, img_rec)
            ssim_val = compute_ssim(img_gen, img_rec)

            psnr_list.append(psnr)
            ssim_list.append(ssim_val)

        return np.array(psnr_list), np.array(ssim_list)

    except Exception as e:
        print(f"Error processing images in {result_dir}: {e}")
        return None, None

def compute_proportion_above_threshold(values, thresholds):
    """Compute proportion of values above each threshold."""
    proportions = []
    for threshold in thresholds:
        prop = np.sum(values >= threshold) / len(values) * 100
        proportions.append(prop)
    return proportions

def create_threshold_plots(output_dir=None):
    """Create threshold comparison plots."""
    if output_dir is None:
        output_dir = '/home/yzeng/remote/skip_inv/results/ablation_study/analysis'
    base_dir = '/home/yzeng/remote/skip_inv/results/ablation_study'
    outputs_dir = '/home/yzeng/remote/skip_inv/outputs'

    methods = [
        ('afpi/threshold_0.3', 'AFPI-0.3', '#FFD93D', base_dir),
        ('afpi/threshold_0.5', 'AFPI-0.5', '#FF6B6B', base_dir),
        ('afpi/threshold_0.7', 'AFPI-0.7', '#4ECDC4', base_dir),
        ('afpi/threshold_0.9', 'AFPI-0.9', '#45B7D1', base_dir),
        ('fpi', 'FPI-default', '#FFA07A', base_dir),
        ('aidi_gs7', 'AIDI-GS7', '#9B59B6', outputs_dir),
    ]

    # Collect data
    data = {}
    for rel_path, method_name, color, data_base_dir in methods:
        result_dir = os.path.join(data_base_dir, rel_path)
        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} not found")
            continue

        # Compute latent metrics
        init_inv_nlm = compute_init_inv_metrics(result_dir)

        # Compute image metrics
        psnr, ssim_val = compute_image_metrics(result_dir)

        if init_inv_nlm is not None and psnr is not None:
            data[method_name] = {
                'init_inv': init_inv_nlm,
                'psnr': psnr,
                'ssim': ssim_val,
                'color': color
            }

    # Generate thresholds with specified ranges
    # Init↔Inv: 5 to 13
    init_inv_thresholds = np.linspace(5, 13, 50)
    # PSNR: 30 to 55
    psnr_thresholds = np.linspace(30, 55, 50)
    # SSIM: 0.85 to 1
    ssim_thresholds = np.linspace(0.85, 1, 50)

    # Create three figures
    fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
    fig2, ax2 = plt.subplots(figsize=(12, 7), dpi=150)
    fig3, ax3 = plt.subplots(figsize=(12, 7), dpi=150)

    # Collect all proportions for y-axis range calculation
    all_props_init_inv = []
    all_props_psnr = []
    all_props_ssim = []

    # Plot Init↔Inv (with focused x-axis range)
    for method_name, metric_data in data.items():
        proportions = compute_proportion_above_threshold(
            metric_data['init_inv'], init_inv_thresholds
        )
        all_props_init_inv.extend(proportions)
        ax1.plot(init_inv_thresholds, proportions, marker='o', markersize=4,
                linewidth=2.5, label=method_name, color=metric_data['color'], alpha=0.8)

    ax1.set_xlabel('Init↔Inv -log(MSE) Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Proportion of Samples Above Threshold (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Init↔Inv: Threshold Comparison', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.95)
    # Fixed x-axis range: 5 to 13
    ax1.set_xlim(5, 13)
    # Dynamic y-axis based on data
    y_min1 = max(0, min(all_props_init_inv) - 5)
    y_max1 = min(100, max(all_props_init_inv) + 5)
    ax1.set_ylim(y_min1, y_max1)
    plt.tight_layout()

    # Plot PSNR (with smart y-axis range)
    for method_name, metric_data in data.items():
        proportions = compute_proportion_above_threshold(
            metric_data['psnr'], psnr_thresholds
        )
        all_props_psnr.extend(proportions)
        ax2.plot(psnr_thresholds, proportions, marker='o', markersize=4,
                linewidth=2.5, label=method_name, color=metric_data['color'], alpha=0.8)

    ax2.set_xlabel('PSNR Threshold (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proportion of Samples Above Threshold (%)', fontsize=12, fontweight='bold')
    ax2.set_title('PSNR: Threshold Comparison', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10, framealpha=0.95)
    # Fixed x-axis range: 30 to 55
    ax2.set_xlim(30, 55)
    # Dynamic y-axis based on data
    y_min2 = max(0, min(all_props_psnr) - 5)
    y_max2 = min(100, max(all_props_psnr) + 5)
    ax2.set_ylim(y_min2, y_max2)
    plt.tight_layout()

    # Plot SSIM (with smart y-axis range)
    for method_name, metric_data in data.items():
        proportions = compute_proportion_above_threshold(
            metric_data['ssim'], ssim_thresholds
        )
        all_props_ssim.extend(proportions)
        ax3.plot(ssim_thresholds, proportions, marker='o', markersize=4,
                linewidth=2.5, label=method_name, color=metric_data['color'], alpha=0.8)

    ax3.set_xlabel('SSIM Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Proportion of Samples Above Threshold (%)', fontsize=12, fontweight='bold')
    ax3.set_title('SSIM: Threshold Comparison', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=10, framealpha=0.95)
    # Fixed x-axis range: 0.85 to 1
    ax3.set_xlim(0.85, 1)
    # Dynamic y-axis based on data
    y_min3 = max(0, min(all_props_ssim) - 5)
    y_max3 = min(100, max(all_props_ssim) + 5)
    ax3.set_ylim(y_min3, y_max3)
    plt.tight_layout()

    # Save figures
    os.makedirs(output_dir, exist_ok=True)

    init_inv_path = os.path.join(output_dir, 'threshold_init_inv.png')
    psnr_path = os.path.join(output_dir, 'threshold_psnr.png')
    ssim_path = os.path.join(output_dir, 'threshold_ssim.png')

    fig1.savefig(init_inv_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig2.savefig(psnr_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig3.savefig(ssim_path, dpi=150, bbox_inches='tight', facecolor='white')

    print(f"✓ Init↔Inv threshold plot saved to: {init_inv_path}")
    print(f"✓ PSNR threshold plot saved to: {psnr_path}")
    print(f"✓ SSIM threshold plot saved to: {ssim_path}")

    # Print statistics
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON STATISTICS")
    print("="*80 + "\n")

    for method_name, metric_data in sorted(data.items()):
        print(f"\n{method_name}:")
        print("-" * 80)

        # Init↔Inv
        init_inv_vals = metric_data['init_inv']
        print(f"  Init↔Inv:")
        print(f"    Mean: {np.mean(init_inv_vals):.4f} ± {np.std(init_inv_vals):.4f}")
        print(f"    Range: [{np.min(init_inv_vals):.4f}, {np.max(init_inv_vals):.4f}]")

        # PSNR
        psnr_vals = metric_data['psnr']
        print(f"  PSNR:")
        print(f"    Mean: {np.mean(psnr_vals):.4f} ± {np.std(psnr_vals):.4f}")
        print(f"    Range: [{np.min(psnr_vals):.4f}, {np.max(psnr_vals):.4f}]")

        # SSIM
        ssim_vals = metric_data['ssim']
        print(f"  SSIM:")
        print(f"    Mean: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
        print(f"    Range: [{np.min(ssim_vals):.4f}, {np.max(ssim_vals):.4f}]")

    plt.close('all')

if __name__ == '__main__':
    create_threshold_plots()
