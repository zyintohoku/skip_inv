import argparse
import csv
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def read_summary(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sample_id"] = int(row["sample_id"])
            row["seed"] = int(row["seed"])
            row["psnr"] = float(row["psnr"])
            rows.append(row)
    return rows


def load_trace(input_root: Path, sample_id: int, seed: int):
    path = input_root / f"sample_{sample_id:04d}" / f"seed_{seed:06d}" / "cfg_directional_trace.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data, path


def pick_low_high_pairs(rows, sample_ids):
    selected = []
    rows_by_sample = {}
    for row in rows:
        rows_by_sample.setdefault(row["sample_id"], []).append(row)
    for sample_id in sample_ids:
        sample_rows = sorted(rows_by_sample[sample_id], key=lambda row: row["psnr"])
        selected.append(sample_rows[0])
        selected.append(sample_rows[-1])
    return selected


def short_prompt(prompt: str, width: int = 72):
    return "\n".join(textwrap.wrap(prompt, width=width))


def plot_single(row, trace, output_path: Path):
    records = trace["records"]
    steps = [r["step_index"] for r in records]
    values = [r["alignment_cos_u"] for r in records]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(steps, values, color="tab:blue", linewidth=1.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xlabel("DDIM step index")
    ax.set_ylabel(r"$\cos_t^{(u)}$")
    ax.grid(alpha=0.25)
    title = (
        f"sample {row['sample_id']:04d}, seed {row['seed']}, PSNR {row['psnr']:.2f}\n"
        f"{short_prompt(row['prompt'], width=68)}"
    )
    ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(selected_rows, traces, output_path: Path):
    n = len(selected_rows)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 2.65 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, selected_rows):
        trace = traces[(row["sample_id"], row["seed"])]
        records = trace["records"]
        steps = [r["step_index"] for r in records]
        values = [r["alignment_cos_u"] for r in records]
        color = "tab:green" if row["psnr"] >= 50 else "tab:red"
        ax.plot(steps, values, color=color, linewidth=1.7)
        ax.axhline(0.0, color="black", linewidth=0.75, alpha=0.55)
        ax.grid(alpha=0.22)
        ax.set_ylabel(r"$\cos_t^{(u)}$")
        label = f"sample {row['sample_id']:04d}, seed {row['seed']}, PSNR {row['psnr']:.2f}"
        prompt = textwrap.shorten(row["prompt"], width=105, placeholder="...")
        ax.set_title(f"{label}\n{prompt}", fontsize=9.5, pad=9)

    axes[-1].set_xlabel("DDIM step index")
    fig.suptitle(r"Examples of $\cos_t^{(u)}$ over time for sensitive prompts", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/cfg_directional_features_seed_sensitive_top10_analysis/feature_vs_psnr/per_seed_cfg_directional_features_with_psnr.csv",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/cfg_directional_features_seed_sensitive_top10/seed_sensitive",
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
        default="results/cfg_directional_features_seed_sensitive_top10_analysis/alignment_cos_time_examples",
    )
    args = parser.parse_args()

    rows = read_summary(Path(args.summary_csv))
    sample_ids = [int(token.strip()) for token in args.sample_ids.split(",") if token.strip()]
    selected_rows = pick_low_high_pairs(rows, sample_ids)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    traces = {}
    for row in selected_rows:
        trace, _ = load_trace(input_root, row["sample_id"], row["seed"])
        traces[(row["sample_id"], row["seed"])] = trace
        plot_single(
            row,
            trace,
            output_dir / f"sample_{row['sample_id']:04d}_seed_{row['seed']:02d}_psnr_{row['psnr']:.2f}_cos_t.png",
        )

    plot_overview(selected_rows, traces, output_dir / "alignment_cos_u_time_examples_overview.png")

    with (output_dir / "selected_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "seed", "psnr", "prompt"])
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "seed": row["seed"],
                    "psnr": f"{row['psnr']:.6f}",
                    "prompt": row["prompt"],
                }
            )

    print(f"saved {len(selected_rows)} individual plots and one overview to {output_dir}")


if __name__ == "__main__":
    main()
