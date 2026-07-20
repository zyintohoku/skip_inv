import argparse
import csv
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch


def read_psnr(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "seed": int(row["seed"]),
                    "psnr": float(row["psnr"]),
                }
            )
    return rows


def load_trace(input_root: Path, sample_id: int, seed: int):
    trace_path = input_root / f"sample_{sample_id:04d}" / f"seed_{seed:06d}" / "prompt_pressure_trace.json"
    tensor_path = trace_path.with_name("trace_tensors.pt")
    with trace_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    tensors = torch.load(tensor_path, map_location="cpu")
    return meta, tensors


def latent_step_lengths(latent_trace: torch.Tensor):
    diffs = latent_trace[1:].float() - latent_trace[:-1].float()
    return torch.linalg.vector_norm(diffs.flatten(start_dim=1), dim=1).cpu().tolist()


def pick_low_high(psnr_rows, input_root: Path, sample_ids):
    by_sample = defaultdict(list)
    for row in psnr_rows:
        trace_path = input_root / f"sample_{row['sample_id']:04d}" / f"seed_{row['seed']:06d}" / "prompt_pressure_trace.json"
        if trace_path.exists():
            by_sample[row["sample_id"]].append(row)

    selected = []
    for sample_id in sample_ids:
        rows = sorted(by_sample[sample_id], key=lambda row: row["psnr"])
        selected.append(rows[0])
        selected.append(rows[-1])
    return selected


def plot_single(row, meta, tensors, output_path: Path):
    timesteps = [record["timestep"] for record in meta["records"]]
    guided = latent_step_lengths(tensors["text_latent_trace"])
    uncond = latent_step_lengths(tensors["uncond_latent_trace"])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(timesteps, guided, color="tab:blue", linewidth=1.9, label="guided")
    ax.plot(timesteps, uncond, color="0.45", linewidth=1.4, linestyle="--", label="unconditional")
    ax.invert_xaxis()
    ax.set_xlabel("DDIM timestep")
    ax.set_ylabel(r"$\|z_{t-1} - z_t\|_2$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    title = (
        f"sample {row['sample_id']:04d}, seed {row['seed']}, PSNR {row['psnr']:.2f}\n"
        f"{textwrap.fill(meta['prompt'], width=72)}"
    )
    ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(selected_rows, traces, output_path: Path):
    fig, axes = plt.subplots(len(selected_rows), 1, figsize=(8.6, 2.65 * len(selected_rows)), sharex=True, sharey=True)
    if len(selected_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, selected_rows):
        meta, tensors = traces[(row["sample_id"], row["seed"])]
        timesteps = [record["timestep"] for record in meta["records"]]
        guided = latent_step_lengths(tensors["text_latent_trace"])
        uncond = latent_step_lengths(tensors["uncond_latent_trace"])
        color = "tab:green" if row["psnr"] >= 50 else "tab:red"

        ax.plot(timesteps, guided, color=color, linewidth=1.8, label="guided")
        ax.plot(timesteps, uncond, color="0.5", linewidth=1.2, linestyle="--", label="unconditional")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.set_ylabel(r"$\|z_{t-1} - z_t\|_2$")
        title = (
            f"sample {row['sample_id']:04d}, seed {row['seed']}, PSNR {row['psnr']:.2f}\n"
            f"{textwrap.shorten(meta['prompt'], width=105, placeholder='...')}"
        )
        ax.set_title(title, fontsize=9.5, pad=9)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("DDIM timestep")
    fig.suptitle(r"Examples of latent step length $\|z_{t-1}-z_t\|_2$ over time", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/prompt_pressure_seed_sensitive_top10/seed_sensitive",
    )
    parser.add_argument(
        "--psnr_detail_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="71,147,218,535",
        help="Comma-separated sample ids; each contributes its lowest-PSNR and highest-PSNR seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/latent_step_length_time_examples",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    sample_ids = [int(token.strip()) for token in args.sample_ids.split(",") if token.strip()]
    selected_rows = pick_low_high(read_psnr(Path(args.psnr_detail_csv)), input_root, sample_ids)

    traces = {}
    for row in selected_rows:
        meta, tensors = load_trace(input_root, row["sample_id"], row["seed"])
        traces[(row["sample_id"], row["seed"])] = (meta, tensors)
        plot_single(
            row,
            meta,
            tensors,
            output_dir / f"sample_{row['sample_id']:04d}_seed_{row['seed']:02d}_psnr_{row['psnr']:.2f}_latent_step_length.png",
        )

    plot_overview(selected_rows, traces, output_dir / "latent_step_length_time_examples_overview.png")

    with (output_dir / "selected_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "seed", "psnr", "prompt"])
        writer.writeheader()
        for row in selected_rows:
            meta, _ = traces[(row["sample_id"], row["seed"])]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "seed": row["seed"],
                    "psnr": f"{row['psnr']:.6f}",
                    "prompt": meta["prompt"],
                }
            )

    print(f"saved {len(selected_rows)} individual plots and one overview to {output_dir}")


if __name__ == "__main__":
    main()
