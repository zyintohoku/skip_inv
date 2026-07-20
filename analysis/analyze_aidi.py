#!/usr/bin/env python3
"""
Analyze AIDI results across different guidance scales (GS1, GS3, GS5, GS7).
Computes latent space metrics: Init↔Inv, Gen↔Rec, Inv↔Gen.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os

def analyze_aidi(outputs_dir='/home/yzeng/remote/skip_inv/outputs'):
    """Analyze AIDI results for all guidance scales."""
    
    results = {}
    
    for gs in [1, 3, 5, 7]:
        result_dir = os.path.join(outputs_dir, f'aidi_gs{gs}')
        
        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} not found")
            continue
        
        # Load latents
        init_latents = torch.load(os.path.join(result_dir, 'init_latents.pt'), map_location='cpu')
        inv_latents = torch.load(os.path.join(result_dir, 'inv_latents.pt'), map_location='cpu')
        gen_latents = torch.load(os.path.join(result_dir, 'gen_latents.pt'), map_location='cpu')
        rec_latents = torch.load(os.path.join(result_dir, 'rec_latents.pt'), map_location='cpu')
        
        # Compute MSE for each sample pair
        init_inv_mse_list = []
        gen_rec_mse_list = []
        inv_gen_mse_list = []
        
        for init, inv, gen, rec in zip(init_latents, inv_latents, gen_latents, rec_latents):
            init_inv_mse_list.append(F.mse_loss(init, inv).item())
            gen_rec_mse_list.append(F.mse_loss(gen, rec).item())
            inv_gen_mse_list.append(F.mse_loss(inv, gen).item())
        
        # Compute mean of -log(MSE) for each metric
        init_inv_nlm = np.mean(-np.log(np.array(init_inv_mse_list)))
        gen_rec_nlm = np.mean(-np.log(np.array(gen_rec_mse_list)))
        inv_gen_nlm = np.mean(-np.log(np.array(inv_gen_mse_list)))
        
        results[f'AIDI-GS{gs}'] = {
            'init_inv_nlm': init_inv_nlm,
            'gen_rec_nlm': gen_rec_nlm,
            'inv_gen_nlm': inv_gen_nlm,
            'init_inv_mse_mean': np.mean(init_inv_mse_list),
            'gen_rec_mse_mean': np.mean(gen_rec_mse_list),
            'inv_gen_mse_mean': np.mean(inv_gen_mse_list),
            'num_samples': len(init_inv_mse_list),
        }
    
    return results

def print_table(results):
    """Print results as markdown table."""
    print()
    print('| Method | Init↔Inv -log(MSE) | Gen↔Rec -log(MSE) | Inv↔Gen -log(MSE) |')
    print('|--------|:------------------:|:------------------:|:------------------:|')
    
    for method in sorted(results.keys()):
        data = results[method]
        print(f"| {method} | {data['init_inv_nlm']:.4f} | {data['gen_rec_nlm']:.4f} | {data['inv_gen_nlm']:.4f} |")
    
    print()
    print('注: -log(MSE) 越大表示 MSE 越小，距离越近')
    print()
    
    # Also print raw MSE for reference
    print('| Method | Init↔Inv MSE | Gen↔Rec MSE | Inv↔Gen MSE | Samples |')
    print('|--------|:------------:|:-----------:|:-----------:|:-------:|')
    
    for method in sorted(results.keys()):
        data = results[method]
        print(f"| {method} | {data['init_inv_mse_mean']:.4f} | {data['gen_rec_mse_mean']:.4f} | {data['inv_gen_mse_mean']:.4f} | {data['num_samples']} |")

def main():
    print("=" * 80)
    print("AIDI Guidance Scale Analysis")
    print("=" * 80)
    
    results = analyze_aidi()
    print_table(results)
    
    print()
    print("=" * 80)
    print("Analysis:")
    print("- Init↔Inv: 反演精度，值越大说明 inv 越接近 init")
    print("- Gen↔Rec: 重建质量，值越大说明 rec 越接近 gen")
    print("- Inv↔Gen: inv 与 gen 的距离，值越大说明越接近")
    print("  (低 GS 时 inv 靠近 gen 而非 init，导致反演失败)")
    print("=" * 80)

if __name__ == '__main__':
    main()
