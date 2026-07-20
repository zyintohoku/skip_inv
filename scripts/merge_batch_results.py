#!/usr/bin/env python3
"""
Merge batch results from multiple batch jobs into a single output.
"""

import torch
import json
import os
import argparse
from pathlib import Path


def merge_batch_results(method='afpi', num_batches=6, output_base='outputs'):
    """Merge results from batch jobs."""
    
    print("="*60)
    print("Merging Batch Results")
    print("="*60)
    print(f"Method: {method}")
    print(f"Number of batches: {num_batches}")
    print("")
    
    # Source directory (where batch files are)
    source_dir = os.path.join(output_base, method)
    
    if not os.path.exists(source_dir):
        print(f"❌ Error: {source_dir} not found!")
        return
    
    # Output directory for merged results
    merged_dir = os.path.join(output_base, f'{method}_merged')
    os.makedirs(merged_dir, exist_ok=True)
    
    # Collect results from all batches
    all_init_latents = []
    all_inv_latents = []
    all_gen_latents = []
    all_rec_latents = []
    all_convergence_losses = []
    
    # Track which batches were found
    found_batches = []
    
    print(f"Looking for batch files in: {source_dir}/")
    print("")
    
    for i in range(num_batches):
        batch_suffix = f'_batch{i}'
        
        print(f"📦 Loading batch {i}...")
        
        # Load latents
        try:
            init_file = os.path.join(source_dir, f'init_latents{batch_suffix}.pt')
            inv_file = os.path.join(source_dir, f'inv_latents{batch_suffix}.pt')
            gen_file = os.path.join(source_dir, f'gen_latents{batch_suffix}.pt')
            rec_file = os.path.join(source_dir, f'rec_latents{batch_suffix}.pt')
            
            if not os.path.exists(init_file):
                print(f"  ⚠ {init_file} not found, skipping")
                continue
            
            init_latents = torch.load(init_file, map_location='cpu')
            inv_latents = torch.load(inv_file, map_location='cpu')
            gen_latents = torch.load(gen_file, map_location='cpu')
            rec_latents = torch.load(rec_file, map_location='cpu')
            
            all_init_latents.extend(init_latents)
            all_inv_latents.extend(inv_latents)
            all_gen_latents.extend(gen_latents)
            all_rec_latents.extend(rec_latents)
            
            print(f"  ✓ Loaded {len(init_latents)} samples")
            found_batches.append(i)
            
        except Exception as e:
            print(f"  ❌ Error loading batch {i}: {e}")
            continue
        
        # Load convergence losses if available
        loss_file = os.path.join(source_dir, f'convergence_losses{batch_suffix}.json')
        if os.path.exists(loss_file):
            try:
                with open(loss_file, 'r') as f:
                    losses = json.load(f)
                all_convergence_losses.extend(losses)
                print(f"  ✓ Loaded convergence losses")
            except Exception as e:
                print(f"  ⚠ Could not load convergence losses: {e}")
    
    if len(found_batches) == 0:
        print("\n❌ No batch results found!")
        return
    
    print("\n" + "="*60)
    print("Saving Merged Results")
    print("="*60)
    
    # Save merged latents
    torch.save(all_init_latents, os.path.join(merged_dir, 'init_latents.pt'))
    torch.save(all_inv_latents, os.path.join(merged_dir, 'inv_latents.pt'))
    torch.save(all_gen_latents, os.path.join(merged_dir, 'gen_latents.pt'))
    torch.save(all_rec_latents, os.path.join(merged_dir, 'rec_latents.pt'))
    
    print(f"✓ Saved {len(all_init_latents)} samples to {merged_dir}/")
    
    # Save merged convergence losses
    if all_convergence_losses:
        # Sort by sample_id
        all_convergence_losses.sort(key=lambda x: x['sample_id'])
        
        with open(os.path.join(merged_dir, 'convergence_losses.json'), 'w') as f:
            json.dump(all_convergence_losses, f, indent=2)
        print(f"✓ Saved convergence losses for {len(all_convergence_losses)} samples")
    
    # Create summary
    summary = {
        'method': method,
        'num_batches': num_batches,
        'found_batches': found_batches,
        'total_samples': len(all_init_latents),
    }
    
    with open(os.path.join(merged_dir, 'merge_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✨ Merge complete!")
    print(f"Results saved to: {merged_dir}/")
    print(f"Batches found: {found_batches}")
    print(f"Total samples: {len(all_init_latents)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="afpi", help="Method name")
    parser.add_argument("--num_batches", type=int, default=6, help="Number of batches")
    parser.add_argument("--output_base", type=str, default="outputs", help="Base output directory")
    args = parser.parse_args()
    
    merge_batch_results(args.method, args.num_batches, args.output_base)


if __name__ == '__main__':
    main()
