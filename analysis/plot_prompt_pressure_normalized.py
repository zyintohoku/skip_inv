import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


METRICS = [
    ("prompt_pressure_P_t", "normalized P_t"),
    ("relative_pressure_R_t", "normalized R_t"),
]


def label_color(label: str):
    if label == "best":
        return "tab:green"
    if label == "worst":
        return "tab:red"
    if label == "seed_sensitive":
        return "tab:purple"
    return None


def load_traces(root: Path):
    traces = []
    for path in sorted(root.glob("*/*/seed_*/prompt_pressure_trace.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        traces.append(
            {
                "label": path.parts[-4],
                "sample_id": int(data["sample_id"]),
                "prompt": data["prompt"],
                "seed": int(data["seed"]),
                "records": data["records"],
            }
        )
    if not traces:
        raise FileNotFoundError(f"No prompt_pressure_trace.json files found under {root}")
    return traces


def normalize(values, mode: str):
    if mode == "max":
        denom = max(values)
    elif mode == "sum":
        denom = sum(values)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    if denom == 0:
        return [0.0 for _ in values]
    return [v / denom for v in values]


def normalized_values(records, metric: str, mode: str):
    values = [r[metric] for r in records]
    return normalize(values, mode)


def mean_curves(curves):
    n_steps = len(curves[0])
    return [sum(curve[i] for curve in curves) / len(curves) for i in range(n_steps)]


def plot_sample_all_seeds(sample_traces, output_path: Path, mode: str):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    prompt = sample_traces[0]["prompt"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        curves = []
        for trace in sample_traces:
            curve = normalized_values(trace["records"], metric, mode)
            curves.append(curve)
            ax.plot(steps, curve, color="tab:blue", alpha=0.20, linewidth=0.8)
        ax.plot(steps, mean_curves(curves), color="black", linewidth=2.0, label="seed mean")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.05 if mode == "max" else None)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"{label} sample {sample_id}: normalized all seeds ({mode})\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_mean_only(sample_traces, output_path: Path, mode: str):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    prompt = sample_traces[0]["prompt"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        curves = [normalized_values(trace["records"], metric, mode) for trace in sample_traces]
        ax.plot(steps, mean_curves(curves), color="black", linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.05 if mode == "max" else None)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(
        f"{label} sample {sample_id}: mean normalized curve over {len(sample_traces)} seeds ({mode})\n{prompt}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_group_sample_means(sample_groups, output_path: Path, mode: str):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for (label, sample_id), traces in sorted(sample_groups.items()):
            color = label_color(label)
            steps = [r["step_index"] for r in traces[0]["records"]]
            curves = [normalized_values(trace["records"], metric, mode) for trace in traces]
            ax.plot(steps, mean_curves(curves), alpha=0.55, linewidth=1.2, color=color)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.05 if mode == "max" else None)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"Per-sample seed-mean normalized curves ({mode})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_top10_analysis/plots_normalized",
    )
    parser.add_argument("--normalization", type=str, choices=["max", "sum"], default="max")
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    output_dir = Path(args.output_dir) / args.normalization
    all_seed_dir = output_dir / "all_seeds"
    mean_dir = output_dir / "seed_mean"
    all_seed_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)

    sample_groups = defaultdict(list)
    for trace in traces:
        sample_groups[(trace["label"], trace["sample_id"])].append(trace)

    for (label, sample_id), sample_traces in sorted(sample_groups.items()):
        plot_sample_all_seeds(
            sample_traces,
            all_seed_dir / f"{label}_sample_{sample_id:04d}_normalized_{args.normalization}_all_seeds.png",
            args.normalization,
        )
        plot_sample_mean_only(
            sample_traces,
            mean_dir / f"{label}_sample_{sample_id:04d}_normalized_{args.normalization}_seed_mean.png",
            args.normalization,
        )

    plot_group_sample_means(
        sample_groups,
        output_dir / f"all_sample_seed_means_normalized_{args.normalization}.png",
        args.normalization,
    )
    print(f"saved normalized plots to {output_dir}")


if __name__ == "__main__":
    main()
