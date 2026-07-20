#!/usr/bin/env python3
"""
Copy worst sample images to analysis directory for easy inspection.
"""

import os
import shutil
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDI_DIR = os.path.join(PROJECT_ROOT, "outputs/reconstruction/aidi_gs7")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/aidi_gs7_analysis")
IMAGES_DIR = os.path.join(RESULTS_DIR, "worst_samples_images")


def copy_sample_images(sample_id, dest_dir):
    """Copy gen and rec images for a sample to destination directory."""
    gen_src = os.path.join(AIDI_DIR, f'{sample_id}gen.png')
    rec_src = os.path.join(AIDI_DIR, f'{sample_id}rec.png')
    
    gen_dst = os.path.join(dest_dir, f'{sample_id}gen.png')
    rec_dst = os.path.join(dest_dir, f'{sample_id}rec.png')
    
    if os.path.exists(gen_src):
        shutil.copy2(gen_src, gen_dst)
    if os.path.exists(rec_src):
        shutil.copy2(rec_src, rec_dst)


def main():
    print("="*60)
    print("Copying Worst Sample Images")
    print("="*60)
    
    # Create main images directory
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Define subdirectories and their source CSV files
    categories = {
        'worst_psnr': 'aidi_gs7_worst_20_psnr.csv',
        'worst_ssim': 'aidi_gs7_worst_20_ssim.csv',
        'worst_lpips': 'aidi_gs7_worst_20_lpips.csv',
        'worst_multi_metrics': 'aidi_gs7_worst_multi_metrics.csv',
        'worst_gen_rec': 'aidi_gs7_worst_50_gen_rec.csv',
    }
    
    for subdir, csv_file in categories.items():
        print(f"\n📁 Processing {subdir}...")
        
        # Create subdirectory
        dest_dir = os.path.join(IMAGES_DIR, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Read CSV
        csv_path = os.path.join(RESULTS_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"   ⚠ {csv_file} not found, skipping")
            continue
        
        df = pd.read_csv(csv_path)
        sample_ids = df['sample_id'].tolist()
        
        print(f"   Copying {len(sample_ids)} samples...")
        
        # Copy images
        for sample_id in sample_ids:
            copy_sample_images(sample_id, dest_dir)
        
        # Count copied files
        copied_files = len([f for f in os.listdir(dest_dir) if f.endswith('.png')])
        print(f"   ✓ Copied {copied_files} images to {subdir}/")
    
    # Create a README
    readme_content = """# Worst Samples Images

This directory contains images for the worst performing samples according to different metrics.

## Subdirectories:

- **worst_psnr/**: Top 20 worst PSNR samples (lowest PSNR)
- **worst_ssim/**: Top 20 worst SSIM samples (lowest SSIM)
- **worst_lpips/**: Top 20 worst LPIPS samples (highest LPIPS)
- **worst_multi_metrics/**: Samples that are worst in 2+ metrics
- **worst_gen_rec/**: Top 50 worst Gen↔Rec latent MSE samples

## File naming:
- `{id}gen.png`: Generated image
- `{id}rec.png`: Reconstructed image

Compare gen.png vs rec.png to visually assess reconstruction quality.
"""
    
    readme_path = os.path.join(IMAGES_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("\n" + "="*60)
    print("✨ Done!")
    print("="*60)
    print(f"Images saved to: {IMAGES_DIR}/")
    print(f"README created: {readme_path}")


if __name__ == '__main__':
    main()
