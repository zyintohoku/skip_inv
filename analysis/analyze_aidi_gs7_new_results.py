#!/usr/bin/env python3
"""
Analyze AIDI-GS7 results from outputs/aidi_gs7: compute metrics and find worst samples.
Computes:
- Init↔Inv -log(MSE)
- Gen↔Rec -log(MSE)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIDI_DIR = os.path.join(PROJECT_ROOT, "outputs/aidi_gs7")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/aidi_gs7_new_analysis")


def compute_metrics():
    """Load latents and compute all metrics per sample."""
    print("Loading latents from:", AIDI_DIR)
    
    init_latents = torch.load(os.path.join(AIDI_DIR, 'init_latents.pt'), map_location='cpu')
    inv_latents = torch.load(os.path.join(AIDI_DIR, 'inv_latents.pt'), map_location='cpu')
    gen_latents = torch.load(os.path.join(AIDI_DIR, 'gen_latents.pt'), map_location='cpu')
    rec_latents = torch.load(os.path.join(AIDI_DIR, 'rec_latents.pt'), map_location='cpu')
    
    print(f"  init_latents: {len(init_latents)} samples")
    print(f"  inv_latents: {len(inv_latents)} samples")
    print(f"  gen_latents: {len(gen_latents)} samples")
    print(f"  rec_latents: {len(rec_latents)} samples")
    
    results = []
    
    for i, (init, inv, gen, rec) in enumerate(zip(init_latents, inv_latents, gen_latents, rec_latents)):
        init_inv_mse = F.mse_loss(init, inv).item()
        gen_rec_mse = F.mse_loss(gen, rec).item()
        
        results.append({
            'sample_id': i,
            'init_inv_mse': init_inv_mse,
            'init_inv_nlm': -np.log(init_inv_mse),
            'gen_rec_mse': gen_rec_mse,
            'gen_rec_nlm': -np.log(gen_rec_mse),
        })
    
    return results


def main():
    print("="*60)
    print("AIDI-GS7 New Results Analysis")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Compute metrics
    results = compute_metrics()
    
    # Compute summary statistics
    init_inv_nlm = [r['init_inv_nlm'] for r in results]
    gen_rec_nlm = [r['gen_rec_nlm'] for r in results]
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total Samples: {len(results)}")
    print(f"Init↔Inv -log(MSE): {np.mean(init_inv_nlm):.4f} ± {np.std(init_inv_nlm):.4f}")
    print(f"Gen↔Rec -log(MSE): {np.mean(gen_rec_nlm):.4f} ± {np.std(gen_rec_nlm):.4f}")
    
    # Save detailed results
    print("\n" + "="*60)
    print("Saving Results...")
    print("="*60)
    
    # Save as JSON
    detailed_json = os.path.join(RESULTS_DIR, 'aidi_gs7_detailed_results.json')
    with open(detailed_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results (JSON): {detailed_json}")
    
    # Save as CSV
    df = pd.DataFrame(results)
    detailed_csv = os.path.join(RESULTS_DIR, 'aidi_gs7_detailed_results.csv')
    df.to_csv(detailed_csv, index=False)
    print(f"✓ Detailed results (CSV): {detailed_csv}")
    
    # Find worst samples for gen-rec
    print("\n" + "="*60)
    print("Worst Samples Analysis")
    print("="*60)
    
    # Sort by gen_rec_nlm (ascending = worst first)
    df_sorted = df.sort_values('gen_rec_nlm')
    worst_50 = df_sorted.head(50)
    
    print(f"\n🔴 Top 50 Worst Gen↔Rec -log(MSE):")
    print(worst_50[['sample_id', 'gen_rec_nlm', 'gen_rec_mse', 'init_inv_nlm']].to_string(index=False))
    
    # Save worst samples
    worst_50_csv = os.path.join(RESULTS_DIR, 'aidi_gs7_worst_50_gen_rec.csv')
    worst_50.to_csv(worst_50_csv, index=False)
    print(f"\n✓ Worst 50 samples saved to: {worst_50_csv}")
    
    # Also find worst for init-inv
    worst_init_inv = df.nsmallest(50, 'init_inv_nlm')
    worst_init_inv_csv = os.path.join(RESULTS_DIR, 'aidi_gs7_worst_50_init_inv.csv')
    worst_init_inv.to_csv(worst_init_inv_csv, index=False)
    print(f"✓ Worst 50 Init↔Inv samples saved to: {worst_init_inv_csv}")
    
    # Summary statistics
    summary = {
        'method': 'aidi_gs7',
        'n_samples': len(results),
        'init_inv_mean': np.mean(init_inv_nlm),
        'init_inv_std': np.std(init_inv_nlm),
        'init_inv_min': np.min(init_inv_nlm),
        'init_inv_max': np.max(init_inv_nlm),
        'gen_rec_mean': np.mean(gen_rec_nlm),
        'gen_rec_std': np.std(gen_rec_nlm),
        'gen_rec_min': np.min(gen_rec_nlm),
        'gen_rec_max': np.max(gen_rec_nlm),
    }
    
    summary_json = os.path.join(RESULTS_DIR, 'aidi_gs7_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics: {summary_json}")
    
    print("\n" + "="*60)
    print("✨ Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
