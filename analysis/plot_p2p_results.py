#!/usr/bin/env python3
"""
Generate plots from saved per-sample P2P evaluation data.
No need to recompute metrics - just loads from p2p_per_sample.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

METHODS = ["aidi_gs1", "aidi_gs3", "aidi_gs5", "aidi_gs7", "p2p_upper_bound"]
METHOD_COLORS = {
    "aidi_gs1": "#3498DB",
    "aidi_gs3": "#2ECC71",
    "aidi_gs5": "#F39C12",
    "aidi_gs7": "#9B59B6",
    "p2p_upper_bound": "#E74C3C",
}
METHOD_LABELS = {
    "aidi_gs1": "AIDI GS1",
    "aidi_gs3": "AIDI GS3",
    "aidi_gs5": "AIDI GS5",
    "aidi_gs7": "AIDI GS7",
    "p2p_upper_bound": "Upper Bound",
}


def load_per_sample_data():
    """Load per-sample metrics from JSON"""
    path = os.path.join(RESULTS_DIR, "p2p_per_sample.json")
    with open(path, 'r') as f:
        return json.load(f)


def plot_clip_threshold_curves(all_samples):
    """Plot CLIP score threshold curves - split into two ranges"""
    
    # Compute data for all methods
    method_data = {}
    for method in METHODS:
        if method not in all_samples:
            continue
        samples = all_samples[method]
        clip_scores = np.array([s["clip_score"] for s in samples])
        method_data[method] = clip_scores
    
    # Plot 1: Low threshold range (0.15 - 0.30)
    plt.figure(figsize=(10, 6))
    thresholds_low = np.linspace(0.15, 0.30, 100)
    
    for method in METHODS:
        if method not in method_data:
            continue
        clip_scores = method_data[method]
        proportions = [np.mean(clip_scores >= t) for t in thresholds_low]
        plt.plot(thresholds_low, proportions, 
                color=METHOD_COLORS[method], 
                label=METHOD_LABELS[method],
                linewidth=2)
    
    # Calculate appropriate y range
    all_props_low = []
    for method in method_data:
        clip_scores = method_data[method]
        props = [np.mean(clip_scores >= t) for t in thresholds_low]
        all_props_low.extend(props)
    y_min_low = max(0, min(all_props_low) - 0.02)
    y_max_low = min(1.0, max(all_props_low) + 0.02)
    
    plt.xlabel("CLIP Score Threshold", fontsize=12)
    plt.ylabel("Proportion of Samples Above Threshold", fontsize=12)
    plt.title("CLIP Score Distribution (Low Threshold: 0.15-0.30)", fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.15, 0.30)
    plt.ylim(y_min_low, y_max_low)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "clip_threshold_low.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: clip_threshold_low.png")
    
    # Plot 2: High threshold range (0.30 - 0.45)
    plt.figure(figsize=(10, 6))
    thresholds_high = np.linspace(0.30, 0.45, 100)
    
    for method in METHODS:
        if method not in method_data:
            continue
        clip_scores = method_data[method]
        proportions = [np.mean(clip_scores >= t) for t in thresholds_high]
        plt.plot(thresholds_high, proportions, 
                color=METHOD_COLORS[method], 
                label=METHOD_LABELS[method],
                linewidth=2)
    
    # Calculate appropriate y range
    all_props_high = []
    for method in method_data:
        clip_scores = method_data[method]
        props = [np.mean(clip_scores >= t) for t in thresholds_high]
        all_props_high.extend(props)
    y_min_high = max(0, min(all_props_high) - 0.02)
    y_max_high = min(1.0, max(all_props_high) + 0.02)
    
    plt.xlabel("CLIP Score Threshold", fontsize=12)
    plt.ylabel("Proportion of Samples Above Threshold", fontsize=12)
    plt.title("CLIP Score Distribution (High Threshold: 0.30-0.45)", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.30, 0.45)
    plt.ylim(y_min_high, y_max_high)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "clip_threshold_high.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: clip_threshold_high.png")


def main():
    print("Loading per-sample data...")
    all_samples = load_per_sample_data()
    
    print("Generating CLIP threshold curves...")
    plot_clip_threshold_curves(all_samples)
    
    print("Done!")


if __name__ == "__main__":
    main()
