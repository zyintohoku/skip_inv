import argparse
import csv
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def load_psnr_detail(path: Path):
    values = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values[(int(row["sample_id"]), int(row["seed"]))] = float(row["psnr"])
    return values


def load_traces(input_root: Path):
    traces = []
    for path in sorted(input_root.glob("sample_*/seed_*/prompt_pressure_trace.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        traces.append(
            {
                "sample_id": int(data["sample_id"]),
                "seed": int(data["seed"]),
                "prompt": data["prompt"],
                "records": data["records"],
                "path": str(path),
            }
        )
    if not traces:
        raise FileNotFoundError(f"No prompt_pressure_trace.json files found under {input_root}")
    return traces


def cumulative(values):
    total = 0.0
    out = []
    for value in values:
        total += value
        out.append(total)
    return out


def plot_sample(sample_id, traces, psnr_by_key, norm, cmap, output_path: Path):
    prompt = traces[0]["prompt"].replace("[", "").replace("]", "")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for trace in sorted(traces, key=lambda item: item["seed"]):
        key = (trace["sample_id"], trace["seed"])
        if key not in psnr_by_key:
            continue
        records = trace["records"]
        x = [int(row["timestep"]) for row in records]
        y = cumulative([float(row["prompt_pressure_P_t"]) for row in records])
        psnr = psnr_by_key[key]
        ax.plot(
            x,
            y,
            color=cmap(norm(psnr)),
            linewidth=1.9,
            alpha=0.9,
            label=f"seed {trace['seed']} | {psnr:.2f}",
        )

    ax.invert_xaxis()
    ax.set_xlabel("DDIM timestep")
    ax.set_ylabel(r"Cumulative $P_t$ sum")
    ax.grid(alpha=0.25)
    title = f"sensitive | sample {sample_id:04d}\n{textwrap.shorten(prompt, width=88, placeholder='...')}"
    ax.set_title(title, fontsize=11)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=0.82)
    cbar_ax = fig.add_axes([0.86, 0.17, 0.025, 0.68])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Reconstruction PSNR")

    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
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
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/sensitive_cumulative_P_by_psnr",
    )
    args = parser.parse_args()

    traces = load_traces(Path(args.input_root))
    psnr_by_key = load_psnr_detail(Path(args.psnr_detail_csv))

    grouped = defaultdict(list)
    psnr_values = []
    for trace in traces:
        grouped[trace["sample_id"]].append(trace)
        key = (trace["sample_id"], trace["seed"])
        if key in psnr_by_key:
            psnr_values.append(psnr_by_key[key])
    if not psnr_values:
        raise ValueError("No PSNR values found for sensitive traces.")

    norm = Normalize(vmin=min(psnr_values), vmax=max(psnr_values))
    cmap = plt.get_cmap("viridis")
    output_dir = Path(args.output_dir)
    for sample_id, sample_traces in sorted(grouped.items()):
        plot_sample(
            sample_id,
            sample_traces,
            psnr_by_key,
            norm,
            cmap,
            output_dir / f"sample_{sample_id:04d}_cumulative_P_t_by_psnr.png",
        )
    print(f"saved {len(grouped)} cumulative P_t plots to {output_dir}")


if __name__ == "__main__":
    main()
