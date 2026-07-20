import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def load_psnr_detail(path: Path):
    psnr_by_key = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            psnr_by_key[(int(row["sample_id"]), int(row["seed"]))] = float(row["psnr"])
    return psnr_by_key


def load_points(input_root: Path, psnr_by_key):
    rows = []
    for path in sorted(input_root.glob("*/*/seed_*/prompt_pressure_trace.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_id = int(data["sample_id"])
        seed = int(data["seed"])
        psnr = psnr_by_key.get((sample_id, seed))
        if psnr is None:
            continue

        records_by_step = {int(r["step_index"]): r for r in data["records"]}
        if 48 not in records_by_step or 49 not in records_by_step:
            raise ValueError(f"Trace does not contain both step 48 and step 49: {path}")

        rows.append(
            {
                "label": data.get("label", path.parts[-4]),
                "sample_id": sample_id,
                "seed": seed,
                "P_48": float(records_by_step[48]["prompt_pressure_P_t"]),
                "P_49": float(records_by_step[49]["prompt_pressure_P_t"]),
                "R_49": float(records_by_step[49]["relative_pressure_R_t"]),
                "PSNR": psnr,
                "trace_path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No joined final-step pressure/PSNR rows found under {input_root}")
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(rows, x_key, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sample_ids = sorted({row["sample_id"] for row in rows})
    cmap = plt.get_cmap("tab10")
    color_by_sample = {sample_id: cmap(i % 10) for i, sample_id in enumerate(sample_ids)}

    for sample_id in sample_ids:
        sample_rows = [row for row in rows if row["sample_id"] == sample_id]
        ax.scatter(
            [row[x_key] for row in sample_rows],
            [row["PSNR"] for row in sample_rows],
            s=34,
            alpha=0.82,
            color=color_by_sample[sample_id],
            label=f"{sample_id}",
            edgecolors="none",
        )

    ax.set_xlabel(x_key)
    ax.set_ylabel("Reconstruction PSNR")
    ax.grid(alpha=0.25)
    ax.set_title(f"Sensitive prompts: {x_key} vs reconstruction PSNR", fontsize=11)
    ax.legend(title="sample_id", fontsize=7, title_fontsize=8, ncol=2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/prompt_pressure_seed_sensitive_top10",
    )
    parser.add_argument(
        "--psnr_detail_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/final_step_vs_psnr",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = load_points(Path(args.input_root), load_psnr_detail(Path(args.psnr_detail_csv)))
    write_csv(output_dir / "sensitive_final_step_pressure_vs_psnr_points.csv", rows)
    plot_scatter(rows, "P_48", output_dir / "sensitive_P_48_vs_PSNR.png")
    plot_scatter(rows, "R_49", output_dir / "sensitive_R_49_vs_PSNR.png")

    print(f"saved {len(rows)} points to {output_dir / 'sensitive_final_step_pressure_vs_psnr_points.csv'}")
    print(f"saved plot: {output_dir / 'sensitive_P_48_vs_PSNR.png'}")
    print(f"saved plot: {output_dir / 'sensitive_R_49_vs_PSNR.png'}")


if __name__ == "__main__":
    main()
