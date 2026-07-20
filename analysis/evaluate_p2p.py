#!/usr/bin/env python3
"""
Evaluate P2P editing results with multiple metrics:
- CLIP Score: similarity between edited image and target prompt
- PSNR: pixel-level similarity (edi vs gen)
- SSIM: structural similarity (edi vs gen)
- LPIPS: perceptual similarity (edi vs gen)

Saves per-sample metrics for detailed analysis.
"""

import os
import json
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDITING_DIR = os.path.join(PROJECT_ROOT, "outputs", "editing")
GEN_DIR = os.path.join(PROJECT_ROOT, "outputs", "reconstruction", "gen")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "PIE_bench", "mapping_file.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "p2p_evaluation_skip_inv")

def discover_methods():
    """Discover evaluable skip_inv_dt* method dirs under outputs/editing."""
    methods = []
    if not os.path.exists(EDITING_DIR):
        return methods

    for name in sorted(os.listdir(EDITING_DIR)):
        base_dir = os.path.join(EDITING_DIR, name)
        if not os.path.isdir(base_dir):
            continue

        if name.startswith("skip_inv_dt") and any(f.endswith("edi.png") for f in os.listdir(base_dir)):
            methods.append((name, base_dir))

    return methods


def load_image_tensor(path, device):
    """Load image as tensor for LPIPS/CLIP (normalized to [-1, 1])"""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2 - 1
    return img.to(device)


def load_image_clip(path, preprocess, device):
    """Load image for CLIP"""
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def load_image_np(path):
    """Load image as numpy array for PSNR/SSIM"""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_clip_score(image_tensor, text_features, clip_model):
    """Compute CLIP similarity score"""
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity


def get_clean_prompt(editing_prompt):
    """Remove brackets from editing prompt"""
    return editing_prompt.replace("[", "").replace("]", "")


def evaluate_method(method_name, method_dir, mapping_data, clip_model, clip_preprocess, lpips_model, device):
    """Evaluate a single method, return per-sample metrics"""
    if not os.path.exists(method_dir):
        print(f"Warning: {method_dir} not found")
        return None
    
    # Per-sample storage
    samples = []
    
    files = os.listdir(method_dir)
    indices = []
    for f in files:
        m = re.match(r"^(\d+)edi\.png$", f)
        if m:
            indices.append(int(m.group(1)))
    indices = sorted(set(indices))
    
    mapping_keys = list(mapping_data.keys())
    
    for idx in tqdm(indices, desc=f"Evaluating {method_name}"):
        edi_path = os.path.join(method_dir, f"{idx}edi.png")
        ori_path = os.path.join(method_dir, f"{idx}ori.png")
        
        if not os.path.exists(edi_path) or not os.path.exists(ori_path):
            continue
        
        if idx >= len(mapping_keys):
            continue
            
        key = mapping_keys[idx]
        editing_prompt = get_clean_prompt(mapping_data[key]["editing_prompt"])
        
        try:
            # CLIP Score
            edi_clip = load_image_clip(edi_path, clip_preprocess, device)
            text_tokens = clip.tokenize([editing_prompt]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            clip_score = compute_clip_score(edi_clip, text_features, clip_model)
            
            # Load images for preservation metrics (edi vs gen)
            edi_np = load_image_np(edi_path)
            gen_path = os.path.join(GEN_DIR, f"{idx}gen.png")
            if not os.path.exists(gen_path):
                gen_path = ori_path
            gen_np = load_image_np(gen_path)
            
            # PSNR
            psnr_val = psnr(gen_np, edi_np, data_range=255)
            
            # SSIM
            ssim_val = ssim(gen_np, edi_np, channel_axis=2, data_range=255)
            
            # LPIPS
            edi_tensor = load_image_tensor(edi_path, device)
            gen_tensor = load_image_tensor(gen_path, device)
            with torch.no_grad():
                lpips_val = lpips_model(edi_tensor, gen_tensor).item()
            
            samples.append({
                "idx": idx,
                "clip_score": float(clip_score),
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "lpips": float(lpips_val),
            })
            
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue
    
    return samples


def compute_summary(samples):
    """Compute mean and std from per-sample data"""
    if not samples:
        return None
    
    clip_scores = [s["clip_score"] for s in samples]
    psnr_scores = [s["psnr"] for s in samples]
    ssim_scores = [s["ssim"] for s in samples]
    lpips_scores = [s["lpips"] for s in samples]
    
    return {
        "clip_score_mean": np.mean(clip_scores),
        "clip_score_std": np.std(clip_scores),
        "psnr_mean": np.mean(psnr_scores),
        "psnr_std": np.std(psnr_scores),
        "ssim_mean": np.mean(ssim_scores),
        "ssim_std": np.std(ssim_scores),
        "lpips_mean": np.mean(lpips_scores),
        "lpips_std": np.std(lpips_scores),
        "n_samples": len(samples),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load models
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Load mapping file
    print("Loading mapping file...")
    with open(MAPPING_FILE, 'r') as f:
        mapping_data = json.load(f)
    
    # Evaluate all methods (per-sample)
    all_samples = {}
    summaries = {}
    method_entries = discover_methods()
    method_order = [name for name, _ in method_entries]
    if not method_entries:
        raise RuntimeError(f"No evaluable method directories found in {EDITING_DIR}")
    
    for method, method_dir in method_entries:
        print(f"\n{'='*50}")
        print(f"Evaluating: {method} ({method_dir})")
        print('='*50)
        samples = evaluate_method(method, method_dir, mapping_data, clip_model, clip_preprocess, lpips_model, device)
        if samples:
            all_samples[method] = samples
            summaries[method] = compute_summary(samples)
    
    # Save per-sample data
    per_sample_path = os.path.join(RESULTS_DIR, "p2p_per_sample.json")
    with open(per_sample_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nPer-sample data saved to: {per_sample_path}")
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "p2p_evaluation.json")
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    # Print results table with std
    print("\n" + "="*100)
    print("P2P Editing Evaluation Results (mean ± std)")
    print("="*100)
    print(f"{'Method':<18} {'CLIP Score':<18} {'PSNR':<18} {'SSIM':<18} {'LPIPS':<18} {'N':<6}")
    print("-"*100)
    
    for method in method_order:
        if method in summaries:
            s = summaries[method]
            print(f"{method:<18} "
                  f"{s['clip_score_mean']:.4f}±{s['clip_score_std']:.4f}  "
                  f"{s['psnr_mean']:.2f}±{s['psnr_std']:.2f}    "
                  f"{s['ssim_mean']:.4f}±{s['ssim_std']:.4f}  "
                  f"{s['lpips_mean']:.4f}±{s['lpips_std']:.4f}  "
                  f"{s['n_samples']:<6}")
    
    print("="*100)
    
    # Save markdown table with std
    output_md = os.path.join(RESULTS_DIR, "p2p_evaluation.md")
    with open(output_md, 'w') as f:
        f.write("# P2P Editing Evaluation Results\n\n")
        f.write("Preservation metrics (PSNR/SSIM/LPIPS) compare **edi vs gen**\n\n")
        f.write("| Method | CLIP Score ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | N |\n")
        f.write("|--------|-------------|--------|--------|---------|---|\n")
        for method in method_order:
            if method in summaries:
                s = summaries[method]
                f.write(f"| {method} | {s['clip_score_mean']:.4f}±{s['clip_score_std']:.4f} | "
                       f"{s['psnr_mean']:.2f}±{s['psnr_std']:.2f} | "
                       f"{s['ssim_mean']:.4f}±{s['ssim_std']:.4f} | "
                       f"{s['lpips_mean']:.4f}±{s['lpips_std']:.4f} | {s['n_samples']} |\n")
        f.write("\n**Note:** Higher CLIP/PSNR/SSIM is better, Lower LPIPS is better\n")
    print(f"Markdown table saved to: {output_md}")
    
if __name__ == "__main__":
    main()
