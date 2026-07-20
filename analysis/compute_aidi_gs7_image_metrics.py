#!/usr/bin/env python3
"""
Compute image quality metrics for AIDI-GS7 reconstruction results.
Metrics: PSNR, SSIM, LPIPS
Compares: gen.png (generated) vs rec.png (reconstructed)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

# Image quality metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LPIPS
import torch
import lpips

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDI_DIR = os.path.join(PROJECT_ROOT, "outputs/reconstruction/aidi_gs7")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/aidi_gs7_analysis")


def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    return psnr(img1, img2, data_range=255)


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    return ssim(img1, img2, channel_axis=2, data_range=255)


def compute_lpips(img1, img2, lpips_model):
    """Compute LPIPS between two images."""
    # Convert to tensor and normalize to [-1, 1]
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    
    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor)
    
    return lpips_value.item()


def main():
    print("="*60)
    print("AIDI-GS7 Image Quality Metrics")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize LPIPS model
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex')
    lpips_model.eval()
    
    # Find all sample IDs
    gen_files = sorted([f for f in os.listdir(AIDI_DIR) if f.endswith('gen.png')])
    sample_ids = [int(f.replace('gen.png', '')) for f in gen_files]
    
    print(f"Found {len(sample_ids)} samples")
    print("Computing metrics...")
    
    results = []
    
    for sample_id in tqdm(sample_ids, desc="Processing"):
        gen_path = os.path.join(AIDI_DIR, f'{sample_id}gen.png')
        rec_path = os.path.join(AIDI_DIR, f'{sample_id}rec.png')
        
        if not os.path.exists(gen_path) or not os.path.exists(rec_path):
            print(f"Warning: Missing files for sample {sample_id}")
            continue
        
        # Load images
        gen_img = load_image(gen_path)
        rec_img = load_image(rec_path)
        
        # Compute metrics
        psnr_value = compute_psnr(gen_img, rec_img)
        ssim_value = compute_ssim(gen_img, rec_img)
        lpips_value = compute_lpips(gen_img, rec_img, lpips_model)
        
        results.append({
            'sample_id': sample_id,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'lpips': lpips_value,
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by sample_id
    df = df.sort_values('sample_id').reset_index(drop=True)
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total Samples: {len(df)}")
    print(f"PSNR: {df['psnr'].mean():.4f} ± {df['psnr'].std():.4f} (higher is better)")
    print(f"SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f} (higher is better)")
    print(f"LPIPS: {df['lpips'].mean():.4f} ± {df['lpips'].std():.4f} (lower is better)")
    
    # Save detailed results
    print("\n" + "="*60)
    print("Saving Results...")
    print("="*60)
    
    # Save as CSV
    csv_path = os.path.join(RESULTS_DIR, 'aidi_gs7_image_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Detailed results (CSV): {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(RESULTS_DIR, 'aidi_gs7_image_metrics.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Detailed results (JSON): {json_path}")
    
    # Summary statistics
    summary = {
        'n_samples': len(df),
        'psnr_mean': float(df['psnr'].mean()),
        'psnr_std': float(df['psnr'].std()),
        'psnr_min': float(df['psnr'].min()),
        'psnr_max': float(df['psnr'].max()),
        'ssim_mean': float(df['ssim'].mean()),
        'ssim_std': float(df['ssim'].std()),
        'ssim_min': float(df['ssim'].min()),
        'ssim_max': float(df['ssim'].max()),
        'lpips_mean': float(df['lpips'].mean()),
        'lpips_std': float(df['lpips'].std()),
        'lpips_min': float(df['lpips'].min()),
        'lpips_max': float(df['lpips'].max()),
    }
    
    summary_path = os.path.join(RESULTS_DIR, 'aidi_gs7_image_metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics: {summary_path}")
    
    # Find worst samples for each metric
    print("\n" + "="*60)
    print("Worst Samples Analysis")
    print("="*60)
    
    # Worst PSNR (lowest 20)
    worst_psnr = df.nsmallest(20, 'psnr')
    print("\n🔴 Top 20 Worst PSNR:")
    print(worst_psnr[['sample_id', 'psnr', 'ssim', 'lpips']].to_string(index=False))
    worst_psnr.to_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_psnr.csv'), index=False)
    
    # Worst SSIM (lowest 20)
    worst_ssim = df.nsmallest(20, 'ssim')
    print("\n🔴 Top 20 Worst SSIM:")
    print(worst_ssim[['sample_id', 'psnr', 'ssim', 'lpips']].to_string(index=False))
    worst_ssim.to_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_ssim.csv'), index=False)
    
    # Worst LPIPS (highest 20)
    worst_lpips = df.nlargest(20, 'lpips')
    print("\n🔴 Top 20 Worst LPIPS:")
    print(worst_lpips[['sample_id', 'psnr', 'ssim', 'lpips']].to_string(index=False))
    worst_lpips.to_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_lpips.csv'), index=False)
    
    # Find samples that are worst across multiple metrics
    print("\n" + "="*60)
    print("Cross-Metric Analysis")
    print("="*60)
    
    # Samples in worst 50 for at least 2 metrics
    worst_psnr_50 = set(df.nsmallest(50, 'psnr')['sample_id'])
    worst_ssim_50 = set(df.nsmallest(50, 'ssim')['sample_id'])
    worst_lpips_50 = set(df.nlargest(50, 'lpips')['sample_id'])
    
    # Samples bad in 2+ metrics
    bad_2_metrics = []
    for sid in df['sample_id']:
        count = sum([
            sid in worst_psnr_50,
            sid in worst_ssim_50,
            sid in worst_lpips_50
        ])
        if count >= 2:
            bad_2_metrics.append(sid)
    
    print(f"\n🔴 {len(bad_2_metrics)} samples are in worst 50 for 2+ metrics:")
    bad_df = df[df['sample_id'].isin(bad_2_metrics)].sort_values('psnr')
    print(bad_df[['sample_id', 'psnr', 'ssim', 'lpips']].to_string(index=False))
    bad_df.to_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_multi_metrics.csv'), index=False)
    
    print("\n" + "="*60)
    print("✨ Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
