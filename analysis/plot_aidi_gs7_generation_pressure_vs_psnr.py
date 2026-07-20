import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SINGLE_PANEL_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def load_joined_rows(pressure_root: Path, psnr_csv: Path):
    psnr_by_key = {
        (int(row["seed"]), int(row["sample_id"])): row
        for row in read_csv_rows(psnr_csv)
    }

    rows = []
    for metrics_path in sorted(pressure_root.glob("seed_*/per_sample_prompt_pressure_metrics.csv")):
        for row in read_csv_rows(metrics_path):
            seed = int(row["seed"])
            sample_id = int(row["sample_id"])
            psnr_row = psnr_by_key.get((seed, sample_id))
            if psnr_row is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "mapping_key": row["mapping_key"],
                    "prompt": row["prompt"],
                    "P_t_sum": float(row["P_raw_sum"]),
                    "R_t_sum": float(row["R_raw_sum"]),
                    "P_entropy": float(row["P_entropy"]),
                    "P_gini": float(row["P_gini"]),
                    "R_entropy": float(row["R_entropy"]),
                    "R_gini": float(row["R_gini"]),
                    "PSNR": float(psnr_row["psnr"]),
                    "MSE": float(psnr_row["mse"]),
                    "gen_path": psnr_row["gen_path"],
                    "rec_path": psnr_row["rec_path"],
                    "gen_mse_to_reference": row.get("gen_mse_to_reference", ""),
                }
            )
    return sorted(rows, key=lambda row: (row["seed"], row["sample_id"]))


def plot_two_panel(rows, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, x_key, color in [
        (axes[0], "P_t_sum", "tab:blue"),
        (axes[1], "R_t_sum", "tab:orange"),
    ]:
        xs = [row[x_key] for row in rows]
        ys = [row["PSNR"] for row in rows]
        corr = pearson(xs, ys)
        ax.scatter(xs, ys, s=18, alpha=0.55, color=color, edgecolors="none")
        ax.set_xlabel(x_key)
        ax.set_title(f"{x_key} vs PSNR (r={corr:.3f})")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("PSNR")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_single_pressure_panel(
    rows,
    x_key: str,
    output_path: Path,
    title: str,
    color: str = "tab:blue",
):
    xs = [row[x_key] for row in rows]
    ys = [row["PSNR"] for row in rows]
    corr = pearson(xs, ys)

    with plt.rc_context(SINGLE_PANEL_STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        ax.scatter(
            xs,
            ys,
            s=54,
            alpha=0.82,
            color=color,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.set_xlabel(x_key, labelpad=8)
        ax.set_ylabel("FPI PSNR", labelpad=8)
        ax.set_title(f"{x_key} vs PSNR (r={corr:.3f})", pad=12)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_metric_scatter(rows, x_key: str, output_path: Path, title: str):
    xs = [row[x_key] for row in rows]
    ys = [row["PSNR"] for row in rows]
    corr = pearson(xs, ys)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=18, alpha=0.55, color="tab:blue", edgecolors="none")
    ax.set_xlabel(x_key)
    ax.set_ylabel("PSNR")
    ax.set_title(f"{x_key} vs PSNR (r={corr:.3f})")
    ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_distribution_metrics(rows, output_dir: Path, title_prefix: str, filename_prefix: str = ""):
    metrics = [
        ("P_entropy", "P_t_entropy"),
        ("P_gini", "P_t_gini"),
        ("R_entropy", "R_t_entropy"),
        ("R_gini", "R_t_gini"),
    ]
    for key, filename_key in metrics:
        output_path = output_dir / f"PSNR_vs_{filename_prefix}{filename_key}.png"
        plot_metric_scatter(
            rows,
            key,
            output_path,
            f"{title_prefix}: {key} vs reconstruction PSNR",
        )


def make_seed_mean_rows(rows):
    by_sample = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    out = []
    for sample_id, sample_rows in sorted(by_sample.items()):
        out.append(
            {
                "sample_id": sample_id,
                "mapping_key": sample_rows[0]["mapping_key"],
                "prompt": sample_rows[0]["prompt"],
                "n": len(sample_rows),
                "P_t_sum": mean([row["P_t_sum"] for row in sample_rows]),
                "R_t_sum": mean([row["R_t_sum"] for row in sample_rows]),
                "P_entropy": mean([row["P_entropy"] for row in sample_rows]),
                "P_gini": mean([row["P_gini"] for row in sample_rows]),
                "R_entropy": mean([row["R_entropy"] for row in sample_rows]),
                "R_gini": mean([row["R_gini"] for row in sample_rows]),
                "PSNR": mean([row["PSNR"] for row in sample_rows]),
                "MSE": mean([row["MSE"] for row in sample_rows]),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pressure_root",
        type=str,
        default="outputs/aidi_gs7_seed_generation_pressure",
    )
    parser.add_argument(
        "--psnr_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/aidi_gs7_seed_generation_pressure_analysis/pressure_vs_psnr",
    )
    parser.add_argument(
        "--title_label",
        type=str,
        default="AIDI",
        help="Dataset/method label used in plot titles.",
    )
    args = parser.parse_args()

    rows = load_joined_rows(Path(args.pressure_root), Path(args.psnr_csv))
    if not rows:
        raise FileNotFoundError(f"No joined rows found under {args.pressure_root}")

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "per_seed_pressure_vs_psnr_points.csv", rows)

    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)

    seed_plot_dir = output_dir / "by_seed"
    for seed, seed_rows in sorted(by_seed.items()):
        plot_two_panel(
            seed_rows,
            seed_plot_dir / f"seed_{seed:02d}_P_R_sum_vs_PSNR.png",
            f"{args.title_label} GS=7 seed {seed}: generation pressure vs reconstruction PSNR",
        )

    mean_rows = make_seed_mean_rows(rows)
    write_csv(output_dir / "seed_mean_pressure_vs_psnr_points.csv", mean_rows)
    plot_two_panel(
        mean_rows,
        output_dir / "seed_mean_P_R_sum_vs_PSNR.png",
        f"{args.title_label} GS=7 seed-mean generation pressure vs mean reconstruction PSNR",
    )
    plot_single_pressure_panel(
        mean_rows,
        "P_t_sum",
        output_dir / "seed_mean_P_t_sum_vs_PSNR.png",
        f"{args.title_label} GS=7 seed-mean generation pressure vs mean reconstruction PSNR",
    )
    plot_distribution_metrics(
        rows,
        output_dir / "entropy_gini",
        f"{args.title_label} GS=7 per-seed generation pressure",
    )
    plot_distribution_metrics(
        mean_rows,
        output_dir / "entropy_gini_seed_mean",
        f"{args.title_label} GS=7 seed-mean generation pressure",
        filename_prefix="mean_",
    )

    print(f"joined rows: {len(rows)}")
    print(f"seed plots: {len(by_seed)} saved to {seed_plot_dir}")
    print(f"seed-mean plot: {output_dir / 'seed_mean_P_R_sum_vs_PSNR.png'}")
    print(f"seed-mean P_t_sum plot: {output_dir / 'seed_mean_P_t_sum_vs_PSNR.png'}")
    print(f"entropy/gini plots: {output_dir / 'entropy_gini'}")
    print(f"seed-mean entropy/gini plots: {output_dir / 'entropy_gini_seed_mean'}")


if __name__ == "__main__":
    main()
