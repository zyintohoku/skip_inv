import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


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


def normalize_sum(values):
    total = sum(values)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def values_from_records(records, normalized=False):
    values = [r["guidance_delta_l2"] for r in records]
    if normalized:
        return normalize_sum(values)
    return values


def mean_curves(curves):
    n_steps = len(curves[0])
    return [sum(curve[i] for curve in curves) / len(curves) for i in range(n_steps)]


def plot_sample(sample_traces, output_path: Path, normalized=False):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    prompt = sample_traces[0]["prompt"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]
    curves = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for trace in sample_traces:
        curve = values_from_records(trace["records"], normalized=normalized)
        curves.append(curve)
        ax.plot(steps, curve, color="tab:blue", alpha=0.20, linewidth=0.8)
    ax.plot(steps, mean_curves(curves), color="black", linewidth=2.0, label="seed mean")
    ax.set_xlabel("DDIM step index")
    ax.set_ylabel("guidance_delta_l2 / sum" if normalized else "guidance_delta_l2")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    title_mode = "sum-normalized" if normalized else "raw"
    fig.suptitle(f"{label} sample {sample_id}: {title_mode} guidance delta\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_all_sample_means(sample_groups, output_path: Path, normalized=False):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (label, sample_id), traces in sorted(sample_groups.items()):
        color = label_color(label)
        steps = [r["step_index"] for r in traces[0]["records"]]
        curves = [values_from_records(trace["records"], normalized=normalized) for trace in traces]
        ax.plot(steps, mean_curves(curves), alpha=0.55, linewidth=1.2, color=color)
    ax.set_xlabel("DDIM step index")
    ax.set_ylabel("guidance_delta_l2 / sum" if normalized else "guidance_delta_l2")
    ax.grid(alpha=0.25)
    title_mode = "sum-normalized" if normalized else "raw"
    fig.suptitle(f"Per-sample seed-mean guidance delta curves ({title_mode})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_label_means(traces, output_path: Path, normalized=False):
    label_groups = defaultdict(list)
    for trace in traces:
        label_groups[trace["label"]].append(trace)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label in sorted(label_groups):
        group = label_groups[label]
        steps = [r["step_index"] for r in group[0]["records"]]
        curves = [values_from_records(trace["records"], normalized=normalized) for trace in group]
        ax.plot(steps, mean_curves(curves), linewidth=2.0, label=label, color=label_color(label))
    ax.set_xlabel("DDIM step index")
    ax.set_ylabel("guidance_delta_l2 / sum" if normalized else "guidance_delta_l2")
    ax.grid(alpha=0.25)
    ax.legend()
    title_mode = "sum-normalized" if normalized else "raw"
    fig.suptitle(f"Label mean guidance delta ({title_mode})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_top10_analysis/guidance_delta_l2_plots",
    )
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    norm_dir = output_dir / "sum_normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    sample_groups = defaultdict(list)
    for trace in traces:
        sample_groups[(trace["label"], trace["sample_id"])].append(trace)

    for (label, sample_id), sample_traces in sorted(sample_groups.items()):
        plot_sample(sample_traces, raw_dir / f"{label}_sample_{sample_id:04d}_guidance_delta_l2.png")
        plot_sample(
            sample_traces,
            norm_dir / f"{label}_sample_{sample_id:04d}_guidance_delta_l2_sum_normalized.png",
            normalized=True,
        )

    plot_all_sample_means(sample_groups, raw_dir / "all_sample_seed_means_guidance_delta_l2.png")
    plot_all_sample_means(
        sample_groups,
        norm_dir / "all_sample_seed_means_guidance_delta_l2_sum_normalized.png",
        normalized=True,
    )
    plot_label_means(traces, raw_dir / "label_mean_guidance_delta_l2.png")
    plot_label_means(traces, norm_dir / "label_mean_guidance_delta_l2_sum_normalized.png", normalized=True)
    print(f"saved guidance delta plots to {output_dir}")


if __name__ == "__main__":
    main()
