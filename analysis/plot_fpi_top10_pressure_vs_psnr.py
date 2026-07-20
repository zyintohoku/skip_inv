import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


LABEL_CONFIG = [
    ("best", "results/fpi_gs7_seed_psnr/prompt_psnr_best30.csv"),
    ("worst", "results/fpi_gs7_seed_psnr/prompt_psnr_worst30.csv"),
    ("seed_sensitive", "results/fpi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv"),
]


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


def load_selected_prompts(label_paths, top_k: int):
    selected = []
    for label, csv_path in label_paths:
        for rank, row in enumerate(read_csv_rows(Path(csv_path))[:top_k], start=1):
            selected.append(
                {
                    "label": label,
                    "rank": rank,
                    "sample_id": int(row["sample_id"]),
                    "mapping_key": row["mapping_key"],
                    "prompt": row["original_prompt"],
                    "psnr_mean": float(row["psnr_mean"]),
                    "psnr_std": float(row["psnr_std"]),
                    "psnr_min": float(row["psnr_min"]),
                    "psnr_max": float(row["psnr_max"]),
                }
            )
    return selected


def load_pressure_by_key(pressure_root: Path):
    pressure_by_key = {}
    for metrics_path in sorted(pressure_root.glob("seed_*/per_sample_prompt_pressure_metrics.csv")):
        for row in read_csv_rows(metrics_path):
            key = (int(row["sample_id"]), int(row["seed"]))
            pressure_by_key[key] = row
    return pressure_by_key


def load_psnr_by_key(psnr_csv: Path):
    return {
        (int(row["sample_id"]), int(row["seed"])): row
        for row in read_csv_rows(psnr_csv)
    }


def merge_rows(selected_prompts, pressure_by_key, psnr_by_key):
    rows = []
    for selected in selected_prompts:
        sample_id = selected["sample_id"]
        for seed in range(1, 11):
            key = (sample_id, seed)
            pressure_row = pressure_by_key.get(key)
            psnr_row = psnr_by_key.get(key)
            if pressure_row is None or psnr_row is None:
                continue
            rows.append(
                {
                    "label": selected["label"],
                    "sample_id": sample_id,
                    "seed": seed,
                    "mapping_key": pressure_row["mapping_key"],
                    "prompt": pressure_row["prompt"],
                    "P_t_sum": float(pressure_row["P_raw_sum"]),
                    "R_t_sum": float(pressure_row["R_raw_sum"]),
                    "Delta_sum": float(pressure_row["Delta_raw_sum"]),
                    "PSNR": float(psnr_row["psnr"]),
                    "MSE": float(psnr_row["mse"]),
                    "gen_path": psnr_row["gen_path"],
                    "rec_path": psnr_row["rec_path"],
                }
            )
    label_order = {label: index for index, (label, _) in enumerate(LABEL_CONFIG)}
    return sorted(rows, key=lambda row: (label_order[row["label"]], row["sample_id"], row["seed"]))


def label_color(label: str):
    if label == "best":
        return "tab:green"
    if label == "worst":
        return "tab:red"
    if label == "seed_sensitive":
        return "tab:purple"
    return "tab:blue"


def display_label(label: str):
    if label == "best":
        return "easy"
    if label == "worst":
        return "hard"
    if label == "seed_sensitive":
        return "intermediate"
    return label


PLOT_STYLE = {
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
}


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_scatter(rows, x_key: str, output_path: Path):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 6.6))
        for label, _ in LABEL_CONFIG:
            label_rows = [row for row in rows if row["label"] == label]
            ax.scatter(
                [row[x_key] for row in label_rows],
                [row["PSNR"] for row in label_rows],
                s=54,
                alpha=0.82,
                color=label_color(label),
                label=display_label(label),
                edgecolors="white",
                linewidths=0.6,
            )

        ax.set_xlabel(x_key, labelpad=8)
        ax.set_ylabel("FPI PSNR", labelpad=8)
        ax.grid(alpha=0.25)
        ax.legend(
            title="FPI label",
            frameon=False,
            loc="best",
        )
        ax.set_title(f"FPI top10 groups: {x_key} vs PSNR", pad=12)
        fig.tight_layout()
        save_figure(fig, output_path)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pressure_root",
        type=str,
        default="outputs/aidi_gs7_seed_generation_pressure",
        help="Root containing seed_*/per_sample_prompt_pressure_metrics.csv.",
    )
    parser.add_argument(
        "--psnr_csv",
        type=str,
        default="results/fpi_gs7_seed_psnr/fpi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fpi_prompt_pressure_saved_latent_top10_merged_analysis/pressure_vs_psnr",
    )
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    selected_prompts = load_selected_prompts(LABEL_CONFIG, args.top_k)
    rows = merge_rows(
        selected_prompts,
        load_pressure_by_key(Path(args.pressure_root)),
        load_psnr_by_key(Path(args.psnr_csv)),
    )
    if not rows:
        raise FileNotFoundError("No joined FPI pressure/PSNR rows found.")

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "selected_prompt_ids.csv", selected_prompts)
    write_csv(output_dir / "pressure_vs_psnr_points.csv", rows)
    write_csv(output_dir / "pressure_vs_psnr_100_points.csv", rows)
    plot_scatter(rows, "P_t_sum", output_dir / "P_t_sum_vs_PSNR.png")
    plot_scatter(rows, "R_t_sum", output_dir / "R_t_sum_vs_PSNR.png")

    print(f"selected prompts: {len(selected_prompts)}")
    print(f"joined rows: {len(rows)}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
