#!/usr/bin/env python3
"""
Visualize worst reconstruction samples from AIDI-GS7 results.
Copies worst sample images to a dedicated folder for easy review.
"""

import os
import shutil
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDI_DIR = os.path.join(PROJECT_ROOT, "outputs/aidi_gs7")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/aidi_gs7_new_analysis")
WORST_IMAGES_DIR = os.path.join(RESULTS_DIR, "worst_samples_images")


def copy_worst_samples():
    """Copy worst sample images to dedicated folder."""
    os.makedirs(WORST_IMAGES_DIR, exist_ok=True)
    
    print("="*60)
    print("Copying Worst Sample Images")
    print("="*60)
    
    # Load worst samples from different metrics
    worst_gen_rec = pd.read_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_50_gen_rec.csv'))
    worst_psnr = pd.read_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_psnr.csv'))
    worst_ssim = pd.read_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_ssim.csv'))
    worst_lpips = pd.read_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_20_lpips.csv'))
    worst_multi = pd.read_csv(os.path.join(RESULTS_DIR, 'aidi_gs7_worst_multi_metrics.csv'))
    
    # Collect all unique worst sample IDs
    worst_ids = set()
    worst_ids.update(worst_gen_rec['sample_id'].head(20).tolist())
    worst_ids.update(worst_psnr['sample_id'].tolist())
    worst_ids.update(worst_ssim['sample_id'].tolist())
    worst_ids.update(worst_lpips['sample_id'].tolist())
    worst_ids.update(worst_multi['sample_id'].tolist())
    
    print(f"Total unique worst samples to copy: {len(worst_ids)}")
    
    copied = 0
    for sample_id in sorted(worst_ids):
        gen_path = os.path.join(AIDI_DIR, f'{sample_id}gen.png')
        rec_path = os.path.join(AIDI_DIR, f'{sample_id}rec.png')
        
        if os.path.exists(gen_path) and os.path.exists(rec_path):
            shutil.copy(gen_path, os.path.join(WORST_IMAGES_DIR, f'{sample_id}gen.png'))
            shutil.copy(rec_path, os.path.join(WORST_IMAGES_DIR, f'{sample_id}rec.png'))
            copied += 1
    
    print(f"✓ Copied {copied} sample pairs to: {WORST_IMAGES_DIR}")
    
    # Create a summary HTML for easy viewing
    create_html_viewer(sorted(worst_ids))


def create_html_viewer(sample_ids):
    """Create an HTML file to view worst samples side-by-side."""
    html_path = os.path.join(WORST_IMAGES_DIR, 'view_worst_samples.html')
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AIDI-GS7 Worst Samples</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .sample { 
            background: white; 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample h3 { margin-top: 0; color: #d32f2f; }
        .images { display: flex; gap: 20px; }
        .image-container { flex: 1; }
        .image-container h4 { margin: 5px 0; font-size: 14px; color: #666; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🔴 AIDI-GS7 Worst Reconstruction Samples</h1>
    <p>Generated vs Reconstructed images for samples with worst metrics</p>
"""
    
    for sample_id in sample_ids:
        html_content += f"""
    <div class="sample">
        <h3>Sample {sample_id}</h3>
        <div class="images">
            <div class="image-container">
                <h4>Generated</h4>
                <img src="{sample_id}gen.png" alt="Generated {sample_id}">
            </div>
            <div class="image-container">
                <h4>Reconstructed</h4>
                <img src="{sample_id}rec.png" alt="Reconstructed {sample_id}">
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created HTML viewer: {html_path}")


def main():
    copy_worst_samples()
    
    print("\n" + "="*60)
    print("✨ Worst Samples Export Complete!")
    print("="*60)
    print(f"View images at: {WORST_IMAGES_DIR}/")
    print(f"Open HTML viewer: {WORST_IMAGES_DIR}/view_worst_samples.html")


if __name__ == '__main__':
    main()
