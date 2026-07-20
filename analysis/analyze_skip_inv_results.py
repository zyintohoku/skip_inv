#!/usr/bin/env python3
"""
Analyze skip_inv results: compute metrics and find worst samples.
Computes:
- Init↔Inv -log(MSE)
- Gen↔Rec -log(MSE)
- Gen↔Rec-Fixed -log(MSE)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def compute_metrics(result_dir):
    """Load latents and compute all metrics per sample."""
    try:
        init_latents = torch.load(os.path.join(result_dir, 'init_latents.pt'), map_location='cpu')
        inv_latents = torch.load(os.path.join(result_dir, 'inv_latents.pt'), map_location='cpu')
        gen_latents = torch.load(os.path.join(result_dir, 'gen_latents.pt'), map_location='cpu')
        rec_latents = torch.load(os.path.join(result_dir, 'rec_latents.pt'), map_location='cpu')
        
        # rec_latents_fixed_cfg might not exist in older runs
        rec_latents_fixed_path = os.path.join(result_dir, 'rec_latents_fixed_cfg.pt')
        if os.path.exists(rec_latents_fixed_path):
            rec_latents_fixed = torch.load(rec_latents_fixed_path, map_location='cpu')
        else:
            rec_latents_fixed = None
            
    except Exception as e:
        print(f"  Error loading latents: {e}")
        return None
    
    results = []
    
    for i, (init, inv, gen, rec) in enumerate(zip(init_latents, inv_latents, gen_latents, rec_latents)):
        init_inv_mse = F.mse_loss(init, inv).item()
        gen_rec_mse = F.mse_loss(gen, rec).item()
        
        sample_result = {
            'sample_id': i,
            'init_inv_mse': init_inv_mse,
            'init_inv_nlm': -np.log(init_inv_mse),
            'gen_rec_mse': gen_rec_mse,
            'gen_rec_nlm': -np.log(gen_rec_mse),
        }
        
        # Add fixed cfg metrics if available
        if rec_latents_fixed is not None and i < len(rec_latents_fixed):
            rec_fixed = rec_latents_fixed[i]
            gen_rec_fixed_mse = F.mse_loss(gen, rec_fixed).item()
            sample_result['gen_rec_fixed_mse'] = gen_rec_fixed_mse
            sample_result['gen_rec_fixed_nlm'] = -np.log(gen_rec_fixed_mse)
        
        results.append(sample_result)
    
    return results


def find_worst_samples(all_results, metric='gen_rec_nlm', top_n=10):
    """Find worst samples across all methods for a given metric."""
    # Collect all samples across methods
    all_samples = []
    for method_name, results in all_results.items():
        for sample in results:
            all_samples.append({
                'method': method_name,
                'sample_id': sample['sample_id'],
                metric: sample.get(metric, float('inf'))
            })
    
    # Sort by metric (lower is worse for -log(MSE))
    df = pd.DataFrame(all_samples)
    worst = df.nsmallest(top_n, metric)
    return worst


def main():
    print("="*60)
    print("Skip_Inv Results Analysis")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Parameters to check
    delta_thresholds = ['5e-9', '1e-9']
    force_converge_steps = [10, 20, 30, 40]
    
    all_results = {}
    summary_data = []
    
    for dt in delta_thresholds:
        for fc in force_converge_steps:
            # Convert dt format for directory name
            dt_short = dt.replace('e-9', 'e9').replace('5e9', 'dt5e9').replace('1e9', 'dt1e9')
            method_name = f'{dt_short}_fc{fc}'
            dir_name = f'skip_inv_{method_name}'
            result_dir = os.path.join(OUTPUTS_DIR, dir_name)
            
            if not os.path.exists(result_dir):
                print(f"\n⚠ Skipping {dir_name}: directory not found")
                continue
            
            print(f"\n📊 Processing {method_name}...")
            results = compute_metrics(result_dir)
            
            if results is None:
                continue
            
            all_results[method_name] = results
            
            # Compute summary statistics
            init_inv_nlm = [r['init_inv_nlm'] for r in results]
            gen_rec_nlm = [r['gen_rec_nlm'] for r in results]
            
            summary = {
                'method': method_name,
                'delta_threshold': dt,
                'force_converge': fc,
                'n_samples': len(results),
                'init_inv_mean': np.mean(init_inv_nlm),
                'init_inv_std': np.std(init_inv_nlm),
                'gen_rec_mean': np.mean(gen_rec_nlm),
                'gen_rec_std': np.std(gen_rec_nlm),
            }
            
            # Add fixed cfg stats if available
            if 'gen_rec_fixed_nlm' in results[0]:
                gen_rec_fixed_nlm = [r['gen_rec_fixed_nlm'] for r in results]
                summary['gen_rec_fixed_mean'] = np.mean(gen_rec_fixed_nlm)
                summary['gen_rec_fixed_std'] = np.std(gen_rec_fixed_nlm)
            
            summary_data.append(summary)
            
            print(f"  Samples: {len(results)}")
            print(f"  Init↔Inv: {summary['init_inv_mean']:.4f} ± {summary['init_inv_std']:.4f}")
            print(f"  Gen↔Rec: {summary['gen_rec_mean']:.4f} ± {summary['gen_rec_std']:.4f}")
            if 'gen_rec_fixed_mean' in summary:
                print(f"  Gen↔Rec-Fixed: {summary['gen_rec_fixed_mean']:.4f} ± {summary['gen_rec_fixed_std']:.4f}")
    
    # Save all results
    print("\n" + "="*60)
    print("Saving Results...")
    print("="*60)
    
    # Save detailed per-sample results
    detailed_output = os.path.join(RESULTS_DIR, 'skip_inv_detailed_results.json')
    with open(detailed_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Detailed results: {detailed_output}")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(RESULTS_DIR, 'skip_inv_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary statistics: {summary_csv}")
    
    # Find worst samples
    print("\n" + "="*60)
    print("Worst Samples Analysis")
    print("="*60)
    
    # Worst for gen-rec
    print("\n🔴 Top 10 Worst Gen↔Rec -log(MSE):")
    worst_gen_rec = find_worst_samples(all_results, 'gen_rec_nlm', 10)
    print(worst_gen_rec.to_string(index=False))
    worst_gen_rec.to_csv(os.path.join(RESULTS_DIR, 'skip_inv_worst_gen_rec.csv'), index=False)
    
    # Worst for gen-rec-fixed (if available)
    if any('gen_rec_fixed_nlm' in r[0] for r in all_results.values() if len(r) > 0):
        print("\n🔴 Top 10 Worst Gen↔Rec-Fixed -log(MSE):")
        worst_gen_rec_fixed = find_worst_samples(all_results, 'gen_rec_fixed_nlm', 10)
        print(worst_gen_rec_fixed.to_string(index=False))
        worst_gen_rec_fixed.to_csv(os.path.join(RESULTS_DIR, 'skip_inv_worst_gen_rec_fixed.csv'), index=False)
    
    # Worst for init-inv
    print("\n🔴 Top 10 Worst Init↔Inv -log(MSE):")
    worst_init_inv = find_worst_samples(all_results, 'init_inv_nlm', 10)
    print(worst_init_inv.to_string(index=False))
    worst_init_inv.to_csv(os.path.join(RESULTS_DIR, 'skip_inv_worst_init_inv.csv'), index=False)
    
    print("\n" + "="*60)
    print("✨ Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
