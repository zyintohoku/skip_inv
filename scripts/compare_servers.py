"""
比较不同服务器上的实验结果
"""
import torch
import argparse
from pathlib import Path
import json
import numpy as np
from scipy import stats

def load_latents(output_dir):
    """加载latent文件"""
    output_dir = Path(output_dir)
    
    results = {}
    for latent_type in ['init', 'inv', 'gen', 'rec']:
        latent_file = output_dir / f"{latent_type}_latents.pt"
        if latent_file.exists():
            results[latent_type] = torch.load(latent_file)
        else:
            print(f"⚠️  Warning: {latent_file} not found")
    
    return results

def compute_metrics(latents1, latents2):
    """计算两组latent之间的差异指标"""
    
    # Init-Inv -log(MSE)
    mse_init_inv1 = torch.mean((latents1['init'] - latents1['inv']) ** 2, dim=[1, 2, 3])
    mse_init_inv2 = torch.mean((latents2['init'] - latents2['inv']) ** 2, dim=[1, 2, 3])
    
    init_inv1 = -torch.log10(mse_init_inv1 + 1e-10)
    init_inv2 = -torch.log10(mse_init_inv2 + 1e-10)
    
    # Gen-Rec -log(MSE)
    mse_gen_rec1 = torch.mean((latents1['gen'] - latents1['rec']) ** 2, dim=[1, 2, 3])
    mse_gen_rec2 = torch.mean((latents2['gen'] - latents2['rec']) ** 2, dim=[1, 2, 3])
    
    gen_rec1 = -torch.log10(mse_gen_rec1 + 1e-10)
    gen_rec2 = -torch.log10(mse_gen_rec2 + 1e-10)
    
    return {
        'init_inv1': init_inv1.numpy(),
        'init_inv2': init_inv2.numpy(),
        'gen_rec1': gen_rec1.numpy(),
        'gen_rec2': gen_rec2.numpy()
    }

def statistical_analysis(metrics):
    """统计分析"""
    
    results = {}
    
    for metric_name in ['init_inv', 'gen_rec']:
        m1 = metrics[f'{metric_name}1']
        m2 = metrics[f'{metric_name}2']
        
        # 基本统计
        diff = m1 - m2
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / (np.abs(m1) + 1e-10) * 100  # 百分比
        
        # t-test
        t_stat, p_value = stats.ttest_rel(m1, m2)
        
        # 相关系数
        correlation = np.corrcoef(m1, m2)[0, 1]
        
        results[metric_name] = {
            'mean_server1': float(np.mean(m1)),
            'mean_server2': float(np.mean(m2)),
            'std_server1': float(np.std(m1)),
            'std_server2': float(np.std(m2)),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_rel_diff_percent': float(np.mean(rel_diff)),
            'max_rel_diff_percent': float(np.max(rel_diff)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'correlation': float(correlation),
            'significant': p_value < 0.05
        }
    
    return results

def print_report(analysis, server1_name, server2_name):
    """打印分析报告"""
    
    print("\n" + "="*80)
    print(f"Server Comparison Report: {server1_name} vs {server2_name}")
    print("="*80)
    
    for metric_name, stats in analysis.items():
        print(f"\n📊 {metric_name.upper()} Metric:")
        print("-" * 80)
        
        print(f"\n  Mean Values:")
        print(f"    {server1_name:<20}: {stats['mean_server1']:>10.6f} (±{stats['std_server1']:.6f})")
        print(f"    {server2_name:<20}: {stats['mean_server2']:>10.6f} (±{stats['std_server2']:.6f})")
        
        print(f"\n  Differences:")
        print(f"    Mean difference      : {stats['mean_diff']:>10.6f}")
        print(f"    Std of differences   : {stats['std_diff']:>10.6f}")
        print(f"    Mean absolute diff   : {stats['mean_abs_diff']:>10.6f}")
        print(f"    Max absolute diff    : {stats['max_abs_diff']:>10.6f}")
        print(f"    Mean relative diff   : {stats['mean_rel_diff_percent']:>10.4f}%")
        print(f"    Max relative diff    : {stats['max_rel_diff_percent']:>10.4f}%")
        
        print(f"\n  Statistical Tests:")
        print(f"    Correlation          : {stats['correlation']:>10.6f}")
        print(f"    t-statistic          : {stats['t_statistic']:>10.6f}")
        print(f"    p-value              : {stats['p_value']:>10.2e}")
        print(f"    Significant (α=0.05) : {'Yes ⚠️' if stats['significant'] else 'No ✅'}")
        
        # 解释
        print(f"\n  Interpretation:")
        if stats['correlation'] > 0.999:
            print(f"    ✅ Extremely high correlation - results are nearly identical")
        elif stats['correlation'] > 0.99:
            print(f"    ✅ Very high correlation - minor numerical differences")
        elif stats['correlation'] > 0.95:
            print(f"    ⚠️  High correlation but noticeable differences")
        else:
            print(f"    ❌ Low correlation - significant differences detected")
        
        if stats['mean_rel_diff_percent'] < 0.01:
            print(f"    ✅ Mean relative difference < 0.01% - negligible")
        elif stats['mean_rel_diff_percent'] < 0.1:
            print(f"    ✅ Mean relative difference < 0.1% - acceptable")
        elif stats['mean_rel_diff_percent'] < 1.0:
            print(f"    ⚠️  Mean relative difference < 1% - minor concern")
        else:
            print(f"    ❌ Mean relative difference > 1% - investigate further")
    
    print("\n" + "="*80)
    print("\n")

def main():
    parser = argparse.ArgumentParser(description='Compare results from different servers')
    parser.add_argument('--dir1', type=str, required=True, help='First output directory')
    parser.add_argument('--dir2', type=str, required=True, help='Second output directory')
    parser.add_argument('--name1', type=str, default='Server 1', help='Name for first server')
    parser.add_argument('--name2', type=str, default='Server 2', help='Name for second server')
    parser.add_argument('--output', type=str, default='server_comparison.json', 
                        help='Output JSON file for detailed results')
    args = parser.parse_args()
    
    print("Loading latents from both servers...")
    latents1 = load_latents(args.dir1)
    latents2 = load_latents(args.dir2)
    
    print("Computing metrics...")
    metrics = compute_metrics(latents1, latents2)
    
    print("Running statistical analysis...")
    analysis = statistical_analysis(metrics)
    
    # 打印报告
    print_report(analysis, args.name1, args.name2)
    
    # 保存详细结果
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Detailed results saved to: {output_file}")
    
    # 总结
    print("\n📋 Summary:")
    all_correlations = [stats['correlation'] for stats in analysis.values()]
    all_rel_diffs = [stats['mean_rel_diff_percent'] for stats in analysis.values()]
    
    print(f"  Average correlation: {np.mean(all_correlations):.6f}")
    print(f"  Average relative difference: {np.mean(all_rel_diffs):.4f}%")
    
    if np.mean(all_correlations) > 0.999 and np.mean(all_rel_diffs) < 0.1:
        print(f"\n  ✅ Conclusion: Results are highly consistent across servers")
        print(f"     Differences are within expected numerical precision limits")
    elif np.mean(all_correlations) > 0.99:
        print(f"\n  ⚠️  Conclusion: Results show minor numerical differences")
        print(f"     This is normal for GPU computations across different hardware")
    else:
        print(f"\n  ❌ Conclusion: Significant differences detected")
        print(f"     Further investigation recommended")

if __name__ == "__main__":
    main()
