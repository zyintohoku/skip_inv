import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


METRICS = [
    ("prompt_pressure_P_t", "P_t"),
    ("relative_pressure_R_t", "R_t"),
    ("bending_B_t", "B_t"),
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


def mean_curve(records_list, metric):
    n_steps = len(records_list[0])
    means = []
    for i in range(n_steps):
        means.append(sum(records[i][metric] for records in records_list) / len(records_list))
    return means


def plot_sample_all_seeds(sample_traces, output_path: Path):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    prompt = sample_traces[0]["prompt"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 9), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for trace in sample_traces:
            values = [r[metric] for r in trace["records"]]
            ax.plot(steps, values, color="tab:blue", alpha=0.18, linewidth=0.8)
        mean_values = mean_curve([t["records"] for t in sample_traces], metric)
        ax.plot(steps, mean_values, color="black", linewidth=2.0, label="seed mean")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"{label} sample {sample_id}: all seeds\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_mean_only(sample_traces, output_path: Path):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    prompt = sample_traces[0]["prompt"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 9), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        mean_values = mean_curve([t["records"] for t in sample_traces], metric)
        ax.plot(steps, mean_values, color="black", linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"{label} sample {sample_id}: mean over {len(sample_traces)} seeds\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_group_sample_means(sample_groups, output_path: Path):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 9), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for (label, sample_id), traces in sorted(sample_groups.items()):
            color = label_color(label)
            steps = [r["step_index"] for r in traces[0]["records"]]
            values = mean_curve([t["records"] for t in traces], metric)
            ax.plot(steps, values, alpha=0.55, linewidth=1.2, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle("Per-sample seed-mean curves", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_top10_analysis/plots_by_sample",
    )
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    output_dir = Path(args.output_dir)
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
            all_seed_dir / f"{label}_sample_{sample_id:04d}_all_seeds.png",
        )
        plot_sample_mean_only(
            sample_traces,
            mean_dir / f"{label}_sample_{sample_id:04d}_seed_mean.png",
        )

    plot_group_sample_means(sample_groups, output_dir / "all_sample_seed_means.png")
    print(f"saved {len(sample_groups)} all-seed plots to {all_seed_dir}")
    print(f"saved {len(sample_groups)} seed-mean plots to {mean_dir}")
    print(f"saved combined sample-mean plot to {output_dir / 'all_sample_seed_means.png'}")


if __name__ == "__main__":
    main()
