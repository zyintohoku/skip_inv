import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch


TRACE_KEYS = [
    ("text_latent_trace", "CFG latent step length"),
    ("uncond_latent_trace", "Uncond latent step length"),
]


def label_color(label: str):
    if label == "best":
        return "tab:green"
    if label == "worst":
        return "tab:red"
    if label == "seed_sensitive":
        return "tab:purple"
    return None


def load_trace_entries(root: Path):
    entries = []
    for json_path in sorted(root.glob("*/*/seed_*/prompt_pressure_trace.json")):
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        tensor_path = json_path.with_name("trace_tensors.pt")
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing tensor trace: {tensor_path}")
        entries.append(
            {
                "label": json_path.parts[-4],
                "sample_id": int(meta["sample_id"]),
                "prompt": meta["prompt"],
                "seed": int(meta["seed"]),
                "tensor_path": tensor_path,
            }
        )
    if not entries:
        raise FileNotFoundError(f"No prompt_pressure_trace.json files found under {root}")
    return entries


def latent_step_lengths(latent_trace: torch.Tensor):
    diffs = latent_trace[1:].float() - latent_trace[:-1].float()
    return torch.linalg.vector_norm(diffs.flatten(start_dim=1), dim=1).cpu().tolist()


def get_lengths(entry):
    tensors = torch.load(entry["tensor_path"], map_location="cpu")
    return {
        key: latent_step_lengths(tensors[key])
        for key, _ in TRACE_KEYS
    }


def attach_lengths(entries):
    loaded = []
    for entry in entries:
        copied = dict(entry)
        copied["lengths"] = get_lengths(entry)
        loaded.append(copied)
    return loaded


def mean_curves(curves):
    n_steps = len(curves[0])
    return [sum(curve[i] for curve in curves) / len(curves) for i in range(n_steps)]


def plot_sample(sample_entries, output_path: Path):
    label = sample_entries[0]["label"]
    sample_id = sample_entries[0]["sample_id"]
    prompt = sample_entries[0]["prompt"]
    sample_entries = sorted(sample_entries, key=lambda x: x["seed"])
    steps = list(range(len(sample_entries[0]["lengths"]["text_latent_trace"])))

    fig, axes = plt.subplots(len(TRACE_KEYS), 1, figsize=(9, 6), sharex=True)
    for ax, (trace_key, ylabel) in zip(axes, TRACE_KEYS):
        curves = []
        for entry in sample_entries:
            curve = entry["lengths"][trace_key]
            curves.append(curve)
            ax.plot(steps, curve, color="tab:blue", alpha=0.20, linewidth=0.8)
        ax.plot(steps, mean_curves(curves), color="black", linewidth=2.0, label="seed mean")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"{label} sample {sample_id}: latent step lengths\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_mean(sample_entries, output_path: Path):
    label = sample_entries[0]["label"]
    sample_id = sample_entries[0]["sample_id"]
    prompt = sample_entries[0]["prompt"]
    sample_entries = sorted(sample_entries, key=lambda x: x["seed"])
    steps = list(range(len(sample_entries[0]["lengths"]["text_latent_trace"])))

    fig, axes = plt.subplots(len(TRACE_KEYS), 1, figsize=(9, 6), sharex=True)
    for ax, (trace_key, ylabel) in zip(axes, TRACE_KEYS):
        curves = [entry["lengths"][trace_key] for entry in sample_entries]
        ax.plot(steps, mean_curves(curves), color="black", linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(f"{label} sample {sample_id}: mean latent step lengths over {len(sample_entries)} seeds\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_all_sample_means(sample_groups, output_path: Path):
    fig, axes = plt.subplots(len(TRACE_KEYS), 1, figsize=(10, 6), sharex=True)
    for ax, (trace_key, ylabel) in zip(axes, TRACE_KEYS):
        for (label, sample_id), entries in sorted(sample_groups.items()):
            color = label_color(label)
            steps = list(range(len(entries[0]["lengths"][trace_key])))
            curves = [entry["lengths"][trace_key] for entry in entries]
            ax.plot(steps, mean_curves(curves), alpha=0.55, linewidth=1.2, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle("Per-sample seed-mean latent step lengths", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_label_means(entries, output_path: Path):
    label_groups = defaultdict(list)
    for entry in entries:
        label_groups[entry["label"]].append(entry)

    fig, axes = plt.subplots(len(TRACE_KEYS), 1, figsize=(10, 6), sharex=True)
    for ax, (trace_key, ylabel) in zip(axes, TRACE_KEYS):
        for label in sorted(label_groups):
            group = label_groups[label]
            steps = list(range(len(group[0]["lengths"][trace_key])))
            curves = [entry["lengths"][trace_key] for entry in group]
            ax.plot(steps, mean_curves(curves), linewidth=2.0, label=label, color=label_color(label))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle("Label mean latent step lengths", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_csv(sample_groups, output_path: Path):
    import csv

    rows = []
    for (label, sample_id), entries in sorted(sample_groups.items()):
        prompt = entries[0]["prompt"]
        per_seed = []
        for entry in entries:
            row = {
                "label": label,
                "sample_id": sample_id,
                "prompt": prompt,
                "seed": entry["seed"],
            }
            for trace_key, _ in TRACE_KEYS:
                vals = entry["lengths"][trace_key]
                row[f"{trace_key}_sum_step_length"] = sum(vals)
                row[f"{trace_key}_mean_step_length"] = sum(vals) / len(vals)
                row[f"{trace_key}_max_step_length"] = max(vals)
                row[f"{trace_key}_max_step"] = max(range(len(vals)), key=lambda i: vals[i])
            per_seed.append(row)
        rows.extend(per_seed)

    if rows:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="outputs/prompt_pressure_top10")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_top10_analysis/latent_step_lengths",
    )
    args = parser.parse_args()

    entries = attach_lengths(load_trace_entries(Path(args.input_root)))
    output_dir = Path(args.output_dir)
    all_seed_dir = output_dir / "all_seeds"
    mean_dir = output_dir / "seed_mean"
    all_seed_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)

    sample_groups = defaultdict(list)
    for entry in entries:
        sample_groups[(entry["label"], entry["sample_id"])].append(entry)

    for (label, sample_id), sample_entries in sorted(sample_groups.items()):
        plot_sample(sample_entries, all_seed_dir / f"{label}_sample_{sample_id:04d}_latent_step_lengths.png")
        plot_sample_mean(sample_entries, mean_dir / f"{label}_sample_{sample_id:04d}_latent_step_lengths_mean.png")

    plot_all_sample_means(sample_groups, output_dir / "all_sample_seed_means_latent_step_lengths.png")
    plot_label_means(entries, output_dir / "label_mean_latent_step_lengths.png")
    write_summary_csv(sample_groups, output_dir / "per_seed_latent_step_lengths.csv")
    print(f"saved latent step length plots to {output_dir}")


if __name__ == "__main__":
    main()
