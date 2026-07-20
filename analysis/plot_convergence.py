#!/usr/bin/env python3
"""
Visualize convergence losses for specific samples.
Plot how the loss changes across timesteps during inversion.
"""

import json
import os
import argparse
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_OUTPUT_BASE = 'outputs/reconstruction'
DEFAULT_METHOD = 'aidi_gs7'
DEFAULT_SAVE_DIR = 'results/aidi_gs7_analysis/worst_samples_images'
DEFAULT_GEN_DIR = os.path.join(DEFAULT_OUTPUT_BASE, 'gen')
DEFAULT_MAPPING_FILE = 'PIE_bench/mapping_file.json'
DEFAULT_TARGET_SAMPLE_IDS = [3, 21, 39]


def resolve_method_dir(method, output_base=DEFAULT_OUTPUT_BASE):
    """Resolve method directory with aidi_gs* fallback when method is not found."""
    method_dir = os.path.join(output_base, method)
    if os.path.isdir(method_dir):
        return method_dir

    if method.startswith('aidi_gs'):
        candidates = sorted(
            d for d in os.listdir(output_base)
            if d.startswith('aidi_gs') and os.path.isdir(os.path.join(output_base, d))
        )
        if candidates:
            return os.path.join(output_base, candidates[-1])

    raise FileNotFoundError(f"Method directory not found: {method_dir}")


def load_convergence_losses(method, output_base=DEFAULT_OUTPUT_BASE):
    """Load convergence losses from method directory JSON files."""
    method_dir = resolve_method_dir(method, output_base)

    # Try merged file first
    merged_file = os.path.join(method_dir, 'convergence_losses.json')
    if os.path.exists(merged_file):
        with open(merged_file, 'r') as f:
            return json.load(f), method_dir

    # Try batch files
    all_losses = []
    batch_idx = 0
    while True:
        batch_file = os.path.join(method_dir, f'convergence_losses_batch{batch_idx}.json')
        if not os.path.exists(batch_file):
            break
        with open(batch_file, 'r') as f:
            all_losses.extend(json.load(f))
        batch_idx += 1

    if all_losses:
        return all_losses, method_dir

    raise FileNotFoundError(f"No convergence_losses files found in {method_dir}")


def load_prompts(mapping_file=DEFAULT_MAPPING_FILE):
    """Load prompt text indexed by sample id from mapping file."""
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    prompts = {}
    for key, value in mapping.items():
        if not isinstance(value, dict):
            continue
        prompt = value.get('original_prompt') or value.get('editing_prompt')
        if prompt is None:
            continue
        try:
            prompts[int(key)] = prompt
        except ValueError:
            continue
    return prompts


def plot_convergence(sample_ids, method=DEFAULT_METHOD, output_base=DEFAULT_OUTPUT_BASE, save_path=None):
    """Plot convergence losses for specified samples."""
    
    # Load convergence losses
    all_losses, method_dir = load_convergence_losses(method, output_base)
    print(f"Loading convergence losses from {method_dir}...")
    
    # Create a dictionary for quick lookup
    losses_dict = {item['sample_id']: item['convergence_losses'] for item in all_losses}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # Colors for different samples
    colors = plt.cm.tab10(np.linspace(0, 1, len(sample_ids)))
    
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in losses_dict:
            print(f"⚠ Warning: Sample {sample_id} not found in convergence losses")
            continue
        
        losses = losses_dict[sample_id]
        timesteps = list(range(len(losses)))
        
        # Plot
        ax.plot(timesteps, losses, marker='o', markersize=4, linewidth=2, 
                label=f'Sample {sample_id}', color=colors[idx], alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convergence Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{method.upper()} - Convergence Loss per Timestep', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    # Use log scale for y-axis if losses span multiple orders of magnitude
    if len(sample_ids) > 0 and sample_ids[0] in losses_dict:
        all_values = []
        for sid in sample_ids:
            if sid in losses_dict:
                all_values.extend(losses_dict[sid])
        if all_values and max(all_values) / min([v for v in all_values if v > 0] or [1]) > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Convergence Loss (log scale)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(sample_id, methods, output_base=DEFAULT_OUTPUT_BASE, save_path=None):
    """Compare convergence losses across different methods for one sample."""
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    for idx, method in enumerate(methods):
        try:
            all_losses, method_dir = load_convergence_losses(method, output_base)
            losses_dict = {item['sample_id']: item['convergence_losses'] for item in all_losses}
            
            if sample_id not in losses_dict:
                print(f"⚠ Warning: Sample {sample_id} not found in {method}")
                continue
            
            losses = losses_dict[sample_id]
            timesteps = list(range(len(losses)))
            
            # Plot
            ax.plot(timesteps, losses, marker='o', markersize=4, linewidth=2, 
                    label=method, color=colors[idx], alpha=0.8)
            
        except Exception as e:
            print(f"⚠ Error loading {method}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convergence Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'Sample {sample_id} - Method Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_with_images(sample_ids, method=DEFAULT_METHOD, output_base=DEFAULT_OUTPUT_BASE, save_path=None, mapping_file=DEFAULT_MAPPING_FILE):
    """Plot convergence curves with corresponding gen/rec images."""
    all_losses, method_dir = load_convergence_losses(method, output_base)
    print(f"Loading convergence losses from {method_dir}...")
    losses_dict = {item['sample_id']: item['convergence_losses'] for item in all_losses}

    prompt_dict = load_prompts(mapping_file)

    valid_samples = [sid for sid in sample_ids if sid in losses_dict]
    missing_samples = [sid for sid in sample_ids if sid not in losses_dict]
    for sid in missing_samples:
        print(f"⚠ Warning: Sample {sid} not found in convergence losses")

    if not valid_samples:
        raise ValueError("No valid sample IDs found in convergence losses.")

    fig, axes = plt.subplots(len(valid_samples), 3, figsize=(16, 4 * len(valid_samples)), dpi=150)
    if len(valid_samples) == 1:
        axes = np.array([axes])

    for row, sample_id in enumerate(valid_samples):
        gen_path = os.path.join(DEFAULT_GEN_DIR, f'{sample_id}gen.png')
        if not os.path.exists(gen_path):
            gen_path = os.path.join(method_dir, f'{sample_id}gen.png')
        rec_path = os.path.join(method_dir, 'gs7', f'{sample_id}rec.png')
        if not os.path.exists(rec_path):
            rec_path = os.path.join(method_dir, f'{sample_id}rec.png')

        ax_gen, ax_rec, ax_curve = axes[row]

        if os.path.exists(gen_path):
            ax_gen.imshow(np.array(Image.open(gen_path).convert('RGB')))
            ax_gen.set_title(f'Sample {sample_id} - Generated', fontsize=11, fontweight='bold')
        else:
            ax_gen.text(0.5, 0.5, f'Missing\n{sample_id}gen.png', ha='center', va='center')
            ax_gen.set_title(f'Sample {sample_id} - Generated', fontsize=11, fontweight='bold')
        ax_gen.axis('off')

        if os.path.exists(rec_path):
            ax_rec.imshow(np.array(Image.open(rec_path).convert('RGB')))
            ax_rec.set_title(f'Sample {sample_id} - Reconstructed', fontsize=11, fontweight='bold')
        else:
            ax_rec.text(0.5, 0.5, f'Missing\n{sample_id}rec.png', ha='center', va='center')
            ax_rec.set_title(f'Sample {sample_id} - Reconstructed', fontsize=11, fontweight='bold')
        ax_rec.axis('off')

        losses = losses_dict[sample_id]
        timesteps = list(range(len(losses)))
        ax_curve.plot(timesteps, losses, marker='o', markersize=4, linewidth=2, color='#1f77b4', alpha=0.9)
        ax_curve.set_xlabel('Timestep', fontsize=10, fontweight='bold')
        ax_curve.set_ylabel('Convergence Loss', fontsize=10, fontweight='bold')
        prompt_text = prompt_dict.get(sample_id, 'Prompt not found')
        wrapped_prompt = textwrap.fill(prompt_text, width=52)
        ax_curve.set_title(
            f'Sample {sample_id} - Convergence\nPrompt: {wrapped_prompt}',
            fontsize=10,
            fontweight='bold'
        )
        ax_curve.grid(True, alpha=0.3, linestyle='--')

        pos_vals = [v for v in losses if v > 0]
        if pos_vals and max(pos_vals) / min(pos_vals) > 100:
            ax_curve.set_yscale('log')
            ax_curve.set_ylabel('Convergence Loss (log scale)', fontsize=10, fontweight='bold')

    fig.suptitle(f'{method.upper()} - Convergence + Images', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize convergence losses')
    parser.add_argument('--sample_ids', type=int, nargs='+', default=DEFAULT_TARGET_SAMPLE_IDS,
                       help='Sample IDs to visualize (default: 3 21 39)')
    parser.add_argument('--method', type=str, default=DEFAULT_METHOD,
                       help='Method name (default: aidi_gs7)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='Multiple methods to compare (for single sample)')
    parser.add_argument('--output_base', type=str, default=DEFAULT_OUTPUT_BASE,
                        help='Base output directory')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--merge_with_images', action='store_true',
                        help='Merge convergence plot with gen/rec images into one figure')
    parser.add_argument('--mapping_file', type=str, default=DEFAULT_MAPPING_FILE,
                        help='Path to mapping_file.json for prompt text')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Convergence Loss Visualization")
    print("="*60)
    
    if args.sample_ids == DEFAULT_TARGET_SAMPLE_IDS and not args.methods:
        args.merge_with_images = True

    if args.save is None:
        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
        if args.methods and len(args.sample_ids) == 1:
            methods_tag = "_".join(args.methods)
            args.save = os.path.join(
                DEFAULT_SAVE_DIR,
                f"convergence_plot_compare_s{args.sample_ids[0]}_{methods_tag}.png"
            )
        else:
            sample_tag = "_".join(str(sid) for sid in args.sample_ids)
            args.save = os.path.join(
                DEFAULT_SAVE_DIR,
                f"convergence_plot_{args.method}_s{sample_tag}.png"
            )

    if args.methods and len(args.sample_ids) == 1:
        # Compare multiple methods for one sample
        print(f"Comparing methods: {args.methods}")
        print(f"Sample ID: {args.sample_ids[0]}")
        plot_comparison(args.sample_ids[0], args.methods, args.output_base, args.save)
    else:
        # Plot multiple samples for one method
        print(f"Method: {args.method}")
        print(f"Sample IDs: {args.sample_ids}")
        if args.merge_with_images:
            plot_convergence_with_images(
                args.sample_ids,
                args.method,
                args.output_base,
                args.save,
                args.mapping_file
            )
        else:
            plot_convergence(args.sample_ids, args.method, args.output_base, args.save)
    
    print("\n✨ Done!")


if __name__ == '__main__':
    main()
