#!/usr/bin/env python3
"""
Generate detailed ablation study comparison report with visualizations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json

def analyze_method(result_dir):
    """Load and analyze latents for a method."""
    try:
        init_latents = torch.load(os.path.join(result_dir, 'init_latents.pt'), map_location='cpu')
        inv_latents = torch.load(os.path.join(result_dir, 'inv_latents.pt'), map_location='cpu')
        gen_latents = torch.load(os.path.join(result_dir, 'gen_latents.pt'), map_location='cpu')
        rec_latents = torch.load(os.path.join(result_dir, 'rec_latents.pt'), map_location='cpu')

        init_inv_mse_list = []
        gen_rec_mse_list = []

        for init, inv, gen, rec in zip(init_latents, inv_latents, gen_latents, rec_latents):
            init_inv_mse = F.mse_loss(init, inv).item()
            gen_rec_mse = F.mse_loss(gen, rec).item()
            init_inv_mse_list.append(init_inv_mse)
            gen_rec_mse_list.append(gen_rec_mse)

        # Compute mean of negative log MSE
        init_inv_nlm = -np.log(np.array(init_inv_mse_list)).mean()
        gen_rec_nlm = -np.log(np.array(gen_rec_mse_list)).mean()

        return {
            'init_vs_inv': init_inv_nlm,
            'gen_vs_rec': gen_rec_nlm,
            'init_inv_mse_list': init_inv_mse_list,
            'gen_rec_mse_list': gen_rec_mse_list,
            'num_samples': len(init_inv_mse_list)
        }
    except Exception as e:
        print(f"Error processing: {e}")
        return None

def get_time_from_log(log_file):
    """Extract time info from log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        total_time = None
        avg_time = None
        for line in lines:
            if 'total_time:' in line:
                total_time = float(line.split(':')[1].strip())
            elif 'avg_time:' in line:
                avg_time = float(line.split(':')[1].strip())
        return total_time, avg_time
    except:
        return None, None

# Configuration
base_dir = '/home/yzeng/remote/skip_inv/results/ablation_study'
outputs_dir = '/home/yzeng/remote/skip_inv/outputs'
log_dir = '/home/yzeng/remote/skip_inv/log'

methods = [
    ('afpi/threshold_0.3', 'AFPI-0.3', 'ablation_afpi_0.3.out', base_dir),
    ('afpi/threshold_0.5', 'AFPI-0.5', 'ablation_afpi_0.5.out', base_dir),
    ('afpi/threshold_0.7', 'AFPI-0.7', 'ablation_afpi_0.7.out', base_dir),
    ('afpi/threshold_0.9', 'AFPI-0.9', 'ablation_afpi_0.9.out', base_dir),
    ('fpi', 'FPI-default', 'ablation_fpi_default.out', base_dir),
    ('aidi_gs7', 'AIDI-GS7', 'aidi_gs7.out', outputs_dir),
]

results = {}

for rel_path, method_name, log_file, data_base_dir in methods:
    result_dir = os.path.join(data_base_dir, rel_path)
    if not os.path.exists(result_dir):
        continue

    analysis = analyze_method(result_dir)
    if analysis:
        total_time, avg_time = get_time_from_log(os.path.join(log_dir, log_file))
        results[method_name] = {
            **analysis,
            'total_time': total_time,
            'avg_time': avg_time,
        }

# Generate markdown report
report = []
report.append("# Ablation Study: Latent Comparison Analysis\n")
report.append("## Summary Table\n")

# Create formatted table
report.append("|Method|Init↔Inv -log(MSE)|Gen↔Rec -log(MSE)|Avg Time (s)|Speed Rank|Quality Rank|")
report.append("|------|----------:|----------:|----------:|--------:|----------:|")

# Sort by different metrics for ranking
by_init_inv = sorted(results.items(), key=lambda x: x[1]['init_vs_inv'], reverse=True)
by_gen_rec = sorted(results.items(), key=lambda x: x[1]['gen_vs_rec'], reverse=True)
by_time = sorted(results.items(), key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] else float('inf'))

init_inv_rank = {m: i+1 for i, (m, _) in enumerate(by_init_inv)}
gen_rec_rank = {m: i+1 for i, (m, _) in enumerate(by_gen_rec)}
time_rank = {m: i+1 for i, (m, _) in enumerate(by_time)}
quality_rank = {m: (init_inv_rank[m] + gen_rec_rank[m]) // 2 for m in results}

for method_name in sorted(results.keys()):
    data = results[method_name]
    avg_time_str = f"{data['avg_time']:.2f}" if data['avg_time'] else 'N/A'
    report.append(
        f"|{method_name}|{data['init_vs_inv']:.6f}|{data['gen_vs_rec']:.6f}|"
        f"{avg_time_str}|"
        f"{time_rank[method_name]}|{quality_rank[method_name]}|"
    )

report.append("\n")
report.append("## Detailed Analysis\n")

# Metric explanations
report.append("### Metrics Definition\n")
report.append("- **Init↔Inv -log(MSE)**: Mean of -log(MSE) between initial and inverted latents\n")
report.append("  - Higher values indicate better inversion accuracy\n")
report.append("  - Range interpretation: -log(x) where x is MSE\n")
report.append("\n- **Gen↔Rec -log(MSE)**: Mean of -log(MSE) between generated and reconstructed latents  \n")
report.append("  - Higher values indicate better reconstruction fidelity\n")
report.append("  - Range interpretation: -log(x) where x is MSE\n")
report.append("\n- **Avg Time**: Average inversion time per sample in seconds\n")
report.append("  - Lower is better (faster inference)\n")
report.append("\n")

# Rankings
report.append("### Rankings by Metric\n")

report.append("**Best Inversion Accuracy (Init↔Inv):**\n")
for rank, (method_name, data) in enumerate(by_init_inv, 1):
    report.append(f"{rank}. {method_name}: {data['init_vs_inv']:.6f}\n")

report.append("\n**Best Reconstruction Quality (Gen↔Rec):**\n")
for rank, (method_name, data) in enumerate(by_gen_rec, 1):
    report.append(f"{rank}. {method_name}: {data['gen_vs_rec']:.6f}\n")

report.append("\n**Fastest Speed (Avg Time):**\n")
for rank, (method_name, data) in enumerate(by_time, 1):
    if data['avg_time']:
        report.append(f"{rank}. {method_name}: {data['avg_time']:.2f}s\n")

report.append("\n**Overall Quality (Average Rank):**\n")
for method, rank in sorted(quality_rank.items(), key=lambda x: x[1]):
    report.append(f"- {method}: Rank {rank}\n")

report.append("\n## Key Findings\n")

best_init_inv = by_init_inv[0]
best_gen_rec = by_gen_rec[0]
best_time = by_time[0]

report.append(f"\n✓ **Best Inversion**: {best_init_inv[0]} with -log(MSE) = {best_init_inv[1]['init_vs_inv']:.6f}\n")
report.append(f"✓ **Best Reconstruction**: {best_gen_rec[0]} with -log(MSE) = {best_gen_rec[1]['gen_vs_rec']:.6f}\n")
report.append(f"✓ **Fastest Method**: {best_time[0]} at {best_time[1]['avg_time']:.2f}s per sample\n")

# Speed vs Quality tradeoff analysis
report.append("\n## Speed vs Quality Tradeoff\n")
report.append("\n| Method | Relative Speed | Quality Score | Tradeoff Score |\n")
report.append("|--------|---------------:|---------------:|---------------:|\n")

fastest_time = by_time[0][1]['avg_time']
for method_name in sorted(results.keys()):
    data = results[method_name]
    rel_speed = (fastest_time / data['avg_time']) * 100 if data['avg_time'] else 0
    quality = (data['init_vs_inv'] + data['gen_vs_rec']) / 2
    tradeoff = quality / (data['avg_time'] / fastest_time) if data['avg_time'] else 0
    report.append(f"|{method_name}|{rel_speed:.1f}%|{quality:.4f}|{tradeoff:.4f}|\n")

report.append("\n")
report.append("**Note**: Relative speed is normalized to the fastest method (100% = fastest)\n")
report.append("Quality score is the average of init↔inv and gen↔rec metrics\n")
report.append("Tradeoff score balances quality and speed\n")

# Test details
report.append("\n## Test Details\n")
report.append(f"\n- **Number of samples tested**: {list(results.values())[0]['num_samples']}\n")
report.append("- **Evaluation metric**: MSE (Mean Squared Error) in latent space\n")
report.append("- **Aggregation method**: Negative log of mean MSE across all samples\n")

# Save report
report_text = ''.join(report)

with open('/home/yzeng/remote/skip_inv/ablation_report.md', 'w') as f:
    f.write(report_text)

print(report_text)

# Save structured results as JSON
json_results = {}
for method_name, data in results.items():
    json_results[method_name] = {
        'init_vs_inv_nlm': float(data['init_vs_inv']),
        'gen_vs_rec_nlm': float(data['gen_vs_rec']),
        'avg_time_s': float(data['avg_time']) if data['avg_time'] else None,
        'total_time_s': float(data['total_time']) if data['total_time'] else None,
        'num_samples': int(data['num_samples']),
        'init_vs_inv_rank': init_inv_rank[method_name],
        'gen_vs_rec_rank': gen_rec_rank[method_name],
        'speed_rank': time_rank[method_name],
    }

with open('/home/yzeng/remote/skip_inv/ablation_results.json', 'w') as f:
    json.dump(json_results, f, indent=2)

print("\n✓ Report saved to: /home/yzeng/remote/skip_inv/ablation_report.md")
print("✓ Results saved to: /home/yzeng/remote/skip_inv/ablation_results.json")
