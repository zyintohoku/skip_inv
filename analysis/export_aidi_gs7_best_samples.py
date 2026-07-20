#!/usr/bin/env python3
"""
Visualize best reconstruction samples from AIDI-GS7 results.
Copies best sample images to a dedicated folder for easy review.
"""

import os
import shutil
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDI_DIR = os.path.join(PROJECT_ROOT, "outputs/reconstruction/aidi_gs7")
GEN_DIR = os.path.join(PROJECT_ROOT, "outputs/reconstruction/gen")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/aidi_gs7_analysis")
BEST_IMAGES_DIR = os.path.join(RESULTS_DIR, "best_samples_images")


def copy_best_samples():
    """Copy best sample images to dedicated folder."""
    os.makedirs(BEST_IMAGES_DIR, exist_ok=True)

    print("=" * 60)
    print("Copying Best Sample Images")
    print("=" * 60)

    # Load per-sample metrics
    detailed = pd.read_csv(os.path.join(RESULTS_DIR, "aidi_gs7_detailed_results.csv"))
    image_metrics = pd.read_csv(os.path.join(RESULTS_DIR, "aidi_gs7_image_metrics.csv"))

    # Best by each metric
    best_gen_rec = detailed.nlargest(50, "gen_rec_nlm")
    best_psnr = image_metrics.nlargest(20, "psnr")
    best_ssim = image_metrics.nlargest(20, "ssim")
    best_lpips = image_metrics.nsmallest(20, "lpips")

    # Find samples that are good across multiple image metrics
    best_psnr_50 = set(image_metrics.nlargest(50, "psnr")["sample_id"])
    best_ssim_50 = set(image_metrics.nlargest(50, "ssim")["sample_id"])
    best_lpips_50 = set(image_metrics.nsmallest(50, "lpips")["sample_id"])
    best_multi_ids = []
    for sid in image_metrics["sample_id"]:
        count = sum([sid in best_psnr_50, sid in best_ssim_50, sid in best_lpips_50])
        if count >= 2:
            best_multi_ids.append(int(sid))
    best_multi = image_metrics[image_metrics["sample_id"].isin(best_multi_ids)].sort_values("psnr", ascending=False)

    # Save CSV summaries
    best_gen_rec.to_csv(os.path.join(RESULTS_DIR, "aidi_gs7_best_50_gen_rec.csv"), index=False)
    best_psnr.to_csv(os.path.join(RESULTS_DIR, "aidi_gs7_best_20_psnr.csv"), index=False)
    best_ssim.to_csv(os.path.join(RESULTS_DIR, "aidi_gs7_best_20_ssim.csv"), index=False)
    best_lpips.to_csv(os.path.join(RESULTS_DIR, "aidi_gs7_best_20_lpips.csv"), index=False)
    best_multi.to_csv(os.path.join(RESULTS_DIR, "aidi_gs7_best_multi_metrics.csv"), index=False)

    # Collect all unique best sample IDs
    best_ids = set()
    best_ids.update(best_gen_rec["sample_id"].head(20).tolist())
    best_ids.update(best_psnr["sample_id"].tolist())
    best_ids.update(best_ssim["sample_id"].tolist())
    best_ids.update(best_lpips["sample_id"].tolist())
    best_ids.update(best_multi["sample_id"].tolist())
    best_ids = sorted(int(x) for x in best_ids)

    print(f"Total unique best samples to copy: {len(best_ids)}")

    copied = 0
    for sample_id in best_ids:
        gen_path = os.path.join(AIDI_DIR, f"{sample_id}gen.png")
        if not os.path.exists(gen_path):
            gen_path = os.path.join(GEN_DIR, f"{sample_id}gen.png")
        rec_path = os.path.join(AIDI_DIR, f"{sample_id}rec.png")
        if os.path.exists(gen_path) and os.path.exists(rec_path):
            shutil.copy(gen_path, os.path.join(BEST_IMAGES_DIR, f"{sample_id}gen.png"))
            shutil.copy(rec_path, os.path.join(BEST_IMAGES_DIR, f"{sample_id}rec.png"))
            copied += 1

    print(f"✓ Copied {copied} sample pairs to: {BEST_IMAGES_DIR}")
    create_html_viewer(best_ids)


def create_html_viewer(sample_ids):
    """Create an HTML file to view best samples side-by-side."""
    html_path = os.path.join(BEST_IMAGES_DIR, "view_best_samples.html")
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AIDI-GS7 Best Samples</title>
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
        .sample h3 { margin-top: 0; color: #2e7d32; }
        .images { display: flex; gap: 20px; }
        .image-container { flex: 1; }
        .image-container h4 { margin: 5px 0; font-size: 14px; color: #666; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🟢 AIDI-GS7 Best Reconstruction Samples</h1>
    <p>Generated vs Reconstructed images for samples with best metrics</p>
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
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"✓ Created HTML viewer: {html_path}")


def main():
    copy_best_samples()
    print("\n" + "=" * 60)
    print("✨ Best Samples Export Complete!")
    print("=" * 60)
    print(f"View images at: {BEST_IMAGES_DIR}/")
    print(f"Open HTML viewer: {BEST_IMAGES_DIR}/view_best_samples.html")


if __name__ == "__main__":
    main()
