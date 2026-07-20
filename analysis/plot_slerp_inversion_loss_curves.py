import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_trace_for_row(row: Dict[str, str]) -> Tuple[List[int], List[float]]:
    image_path = row.get("reconstructed_image_path") or row.get("generated_image_path")
    if not image_path:
        raise ValueError("Row does not contain image paths.")
    trace_path = Path(image_path).parent / "trace_records.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace file: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    records = trace["inversion_records"]
    losses = [float(value) for value in trace["convergence_losses"]]
    if len(records) != len(losses):
        raise ValueError(f"record/loss length mismatch in {trace_path}")

    paired = sorted(
        (
            int(record["timestep"]),
            -math.log10(max(loss, 1e-30)),
        )
        for record, loss in zip(records, losses)
    )
    timesteps = [item[0] for item in paired]
    neg_log_losses = [item[1] for item in paired]
    return timesteps, neg_log_losses


def plot_group(csv_path: Path, output_dir: Path, cmap_name: str) -> None:
    rows = sorted(read_csv(csv_path), key=lambda row: float(row["alpha"]))
    if not rows:
        return

    psnrs = [float(row["gen_rec_image_psnr"]) for row in rows]
    norm = colors.Normalize(vmin=min(psnrs), vmax=max(psnrs))
    cmap = cm.get_cmap(cmap_name)

    group = csv_path.parent.name
    pair_label = rows[0]["pair_label"].replace("_", "-")
    seed_a = rows[0]["seed_a"]
    seed_b = rows[0]["seed_b"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for row in rows:
        alpha = float(row["alpha"])
        psnr = float(row["gen_rec_image_psnr"])
        timesteps, neg_log_losses = load_trace_for_row(row)
        color = cmap(norm(psnr))
        ax.plot(
            timesteps,
            neg_log_losses,
            color=color,
            linewidth=1.8,
            alpha=0.9,
        )
        ax.text(
            timesteps[-1] + 6,
            neg_log_losses[-1],
            f"{alpha:.1f}",
            color=color,
            fontsize=7,
            va="center",
        )

    ax.set_title(f"{group.replace('_', '-')} ({pair_label}, seed {seed_a} to seed {seed_b})")
    ax.set_xlabel("Inversion timestep")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{FPI\ loss})$")
    ax.set_xlim(0, 1040)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Reconstruction PSNR")

    ax.text(
        0.99,
        0.02,
        r"line labels: $\alpha$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{group}_inversion_loss_curves.png", dpi=220)
    fig.savefig(output_dir / f"{group}_inversion_loss_curves.pdf")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    csv_paths = sorted(input_root.glob("*/per_alpha_slerp_fpi_metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per_alpha_slerp_fpi_metrics.csv found under {input_root}")

    for csv_path in csv_paths:
        plot_group(csv_path, output_dir, args.cmap)
    print(f"saved inversion loss curve plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inversion FPI loss curves along t=1->981, colored by reconstruction PSNR."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/slerp_fpi_gs7_seed_sensitive_sample0443",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/slerp_fpi_gs7_seed_sensitive_sample0443_metric_tables/inversion_loss_curves",
    )
    parser.add_argument("--cmap", type=str, default="viridis")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
