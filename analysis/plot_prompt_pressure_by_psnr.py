import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


METRICS = [
    ("prompt_pressure_P_t", "P_t"),
    ("relative_pressure_R_t", "R_t"),
]


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


def load_sample_psnr(path: Path):
    values = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values[int(row["sample_id"])] = float(row["psnr_mean"])
    return values


def load_seed_psnr(path: Path):
    values = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values[(int(row["sample_id"]), int(row["seed"]))] = float(row["psnr"])
    return values


def mean_curve(records_list, metric):
    n_steps = len(records_list[0])
    return [
        sum(records[i][metric] for records in records_list) / len(records_list)
        for i in range(n_steps)
    ]


def plot_sample_mean_metric_colored(sample_groups, psnr_by_sample, metric, ylabel, output_path: Path):
    sample_ids = [sample_id for (_, sample_id) in sample_groups]
    psnr_values = [psnr_by_sample[sample_id] for sample_id in sample_ids if sample_id in psnr_by_sample]
    if not psnr_values:
        raise ValueError("No PSNR values found for plotted samples.")

    norm = Normalize(vmin=min(psnr_values), vmax=max(psnr_values))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for (label, sample_id), traces in sorted(sample_groups.items()):
        if sample_id not in psnr_by_sample:
            continue
        steps = [r["step_index"] for r in traces[0]["records"]]
        values = mean_curve([t["records"] for t in traces], metric)
        color = cmap(norm(psnr_by_sample[sample_id]))
        ax.plot(steps, values, alpha=0.85, linewidth=1.6, color=color)

    ax.set_xlabel("DDIM step index")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.set_title(f"Per-sample seed-mean {ylabel} curves colored by mean PSNR", fontsize=11)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.87, 0.16, 0.025, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("mean PSNR")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def metric_ylims(sample_groups):
    ylims = {}
    for metric, _ in METRICS:
        values = [
            record[metric]
            for traces in sample_groups.values()
            for trace in traces
            for record in trace["records"]
        ]
        y_min = min(values)
        y_max = max(values)
        pad = (y_max - y_min) * 0.04 if y_max > y_min else max(abs(y_max), 1.0) * 0.04
        ylims[metric] = (y_min - pad, y_max + pad)
    return ylims


def plot_sample_all_seeds_metric_colored(
    sample_traces,
    psnr_by_seed,
    norm,
    cmap,
    metric,
    ylabel,
    ylim,
    output_path: Path,
):
    label = sample_traces[0]["label"]
    sample_id = sample_traces[0]["sample_id"]
    sample_traces = sorted(sample_traces, key=lambda x: x["seed"])
    steps = [r["step_index"] for r in sample_traces[0]["records"]]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for trace in sample_traces:
        key = (sample_id, trace["seed"])
        if key not in psnr_by_seed:
            continue
        values = [r[metric] for r in trace["records"]]
        color = cmap(norm(psnr_by_seed[key]))
        ax.plot(steps, values, alpha=0.9, linewidth=1.2, color=color)

    mean_values = mean_curve([t["records"] for t in sample_traces], metric)
    ax.plot(steps, mean_values, color="black", linewidth=2.0, label="seed mean")
    ax.set_xlabel("DDIM step index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title(f"{label} sample {sample_id}: all-seed {ylabel} curves colored by PSNR", fontsize=11)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.87, 0.16, 0.025, 0.72])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("PSNR")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_seed_sensitive_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/components/seed_sensitive_top10/plots_by_sample",
    )
    parser.add_argument(
        "--psnr_by_sample_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_by_sample.csv",
    )
    parser.add_argument(
        "--psnr_detail_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    psnr_by_sample = load_sample_psnr(Path(args.psnr_by_sample_csv))
    psnr_by_seed = load_seed_psnr(Path(args.psnr_detail_csv))

    sample_groups = defaultdict(list)
    for trace in traces:
        sample_groups[(trace["label"], trace["sample_id"])].append(trace)

    for metric, ylabel in METRICS:
        output_path = Path(args.output_dir) / f"all_sample_seed_means_{ylabel}_colored_by_psnr.png"
        plot_sample_mean_metric_colored(sample_groups, psnr_by_sample, metric, ylabel, output_path)
        print(f"saved PSNR-colored plot to {output_path}")

    seed_psnr_values = [
        psnr_by_seed[(sample_id, trace["seed"])]
        for (_, sample_id), sample_traces in sample_groups.items()
        for trace in sample_traces
        if (sample_id, trace["seed"]) in psnr_by_seed
    ]
    if not seed_psnr_values:
        raise ValueError("No seed-level PSNR values found for plotted traces.")
    norm = Normalize(vmin=min(seed_psnr_values), vmax=max(seed_psnr_values))
    cmap = plt.get_cmap("viridis")
    ylims = metric_ylims(sample_groups)
    all_seed_dir = Path(args.output_dir) / "all_seeds_colored_by_psnr"
    for (label, sample_id), sample_traces in sorted(sample_groups.items()):
        for metric, ylabel in METRICS:
            output_path = all_seed_dir / f"{label}_sample_{sample_id:04d}_all_seeds_{ylabel}_colored_by_psnr.png"
            plot_sample_all_seeds_metric_colored(
                sample_traces,
                psnr_by_seed,
                norm,
                cmap,
                metric,
                ylabel,
                ylims[metric],
                output_path,
            )
            print(f"saved PSNR-colored all-seed plot to {output_path}")


if __name__ == "__main__":
    main()
