import argparse
import csv
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


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
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def quantile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "variance": stdev(values) ** 2,
        "std": stdev(values),
        "min": min(values) if values else 0.0,
        "q05": quantile(values, 0.05),
        "q10": quantile(values, 0.10),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.90),
        "q95": quantile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def normalize_perturb_rows(rows):
    out = []
    for row in rows:
        item = dict(row)
        for key in ["sample_id", "seed", "perturb_idx", "num_of_ddim_steps"]:
            if key in item and item[key] != "":
                item[key] = int(float(item[key]))
        for key in [
            "rho",
            "noise_sigma",
            "guidance_scale",
            "psnr",
            "mse",
            "init_delta_l2",
            "final_delta_l2",
            "terminal_sensitivity_S0",
        ]:
            if key in item and item[key] != "":
                item[key] = float(item[key])
        out.append(item)
    return out


def load_perturb_rows(input_root: Path):
    root_csv = input_root / "per_perturb_terminal_sensitivity.csv"
    if root_csv.exists():
        return normalize_perturb_rows(read_csv_rows(root_csv))

    matches = sorted(input_root.glob("*/*/per_perturb_terminal_sensitivity.csv"))
    if not matches:
        matches = sorted(input_root.glob("*/*/seed_*/per_perturb_terminal_sensitivity.csv"))
    if not matches:
        raise FileNotFoundError(f"No per_perturb_terminal_sensitivity.csv found under {input_root}")

    rows = []
    for path in matches:
        rows.extend(read_csv_rows(path))
    return normalize_perturb_rows(rows)


def build_seed_summary(perturb_rows):
    grouped = defaultdict(list)
    for row in perturb_rows:
        grouped[(row["label"], row["sample_id"], row["seed"])].append(row)

    seed_rows = []
    for (_, _, _), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["perturb_idx"])
        base = rows[0]
        values = [row["terminal_sensitivity_S0"] for row in rows]
        init_deltas = [row["init_delta_l2"] for row in rows]
        final_deltas = [row["final_delta_l2"] for row in rows]
        out = {
            "label": base.get("label", ""),
            "sample_id": base["sample_id"],
            "mapping_key": base.get("mapping_key", ""),
            "prompt": base.get("prompt", ""),
            "seed": base["seed"],
            "perturbation_method": base.get("perturbation_method", ""),
            "rho": base.get("rho", ""),
            "noise_sigma": base.get("noise_sigma", ""),
            "guidance_scale": base.get("guidance_scale", ""),
            "num_of_ddim_steps": base.get("num_of_ddim_steps", ""),
            "psnr": base.get("psnr", ""),
            "mse": base.get("mse", ""),
            "gen_path": base.get("gen_path", ""),
            "rec_path": base.get("rec_path", ""),
            "baseline_generated_path": base.get("baseline_generated_path", ""),
        }
        for key, value in summarize(values).items():
            out[f"S0_{key}"] = value
        for key, value in summarize(init_deltas).items():
            out[f"init_delta_l2_{key}"] = value
        for key, value in summarize(final_deltas).items():
            out[f"final_delta_l2_{key}"] = value
        seed_rows.append(out)
    return seed_rows


def build_overall_summary(perturb_rows):
    values = [row["terminal_sensitivity_S0"] for row in perturb_rows]
    init_deltas = [row["init_delta_l2"] for row in perturb_rows]
    final_deltas = [row["final_delta_l2"] for row in perturb_rows]
    out = {}
    for key, value in summarize(values).items():
        out[f"S0_{key}"] = value
    for key, value in summarize(init_deltas).items():
        out[f"init_delta_l2_{key}"] = value
    for key, value in summarize(final_deltas).items():
        out[f"final_delta_l2_{key}"] = value
    return [out]


def seed_sort_key(row):
    psnr = row.get("psnr", "")
    if psnr == "" or psnr is None:
        return (float("inf"), row["seed"])
    return (float(psnr), row["seed"])


def plot_seed_box(perturb_rows, seed_rows, output_dir: Path):
    grouped = defaultdict(list)
    for row in perturb_rows:
        grouped[row["seed"]].append(row["terminal_sensitivity_S0"])
    ordered_seed_rows = sorted(seed_rows, key=seed_sort_key)
    seeds = [row["seed"] for row in ordered_seed_rows]
    values = [grouped[seed] for seed in seeds]
    labels = [
        f"{row['seed']}\n{float(row['psnr']):.2f}" if row.get("psnr", "") != "" else str(row["seed"])
        for row in ordered_seed_rows
    ]

    fig, ax = plt.subplots(figsize=(max(7.2, 0.55 * len(seeds)), 4.8))
    bp = ax.boxplot(values, tick_labels=labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.28)
    ax.set_xlabel("seed\nPSNR")
    ax.set_ylabel(r"terminal sensitivity $S_0$")
    ax.set_title(r"$S_0$ distribution by seed, ordered by PSNR", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "S0_by_seed_box.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_summary_bars(seed_rows, output_dir: Path):
    seed_rows = sorted(seed_rows, key=seed_sort_key)
    labels = [
        f"{row['seed']}\n{float(row['psnr']):.2f}" if row.get("psnr", "") != "" else str(row["seed"])
        for row in seed_rows
    ]
    x = list(range(len(seed_rows)))
    medians = [row["S0_median"] for row in seed_rows]
    q25 = [row["S0_q25"] for row in seed_rows]
    q75 = [row["S0_q75"] for row in seed_rows]
    means = [row["S0_mean"] for row in seed_rows]
    lower = [m - lo for m, lo in zip(medians, q25)]
    upper = [hi - m for m, hi in zip(medians, q75)]

    fig, ax = plt.subplots(figsize=(max(7.2, 0.6 * len(seed_rows)), 4.8))
    ax.bar(x, medians, color="tab:blue", alpha=0.55, label="median")
    ax.errorbar(x, medians, yerr=[lower, upper], fmt="none", ecolor="black", capsize=4, linewidth=1.0, label="IQR")
    ax.scatter(x, means, color="tab:orange", s=32, zorder=3, label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("seed\nPSNR")
    ax.set_ylabel(r"terminal sensitivity $S_0$")
    ax.set_title(r"Per-seed $S_0$ summary, ordered by PSNR", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "S0_seed_summary_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(output_dir: Path, seed_rows, overall_rows):
    overall = overall_rows[0]
    prompt = seed_rows[0].get("prompt", "") if seed_rows else ""
    sample_id = seed_rows[0].get("sample_id", "") if seed_rows else ""
    noise_sigma = seed_rows[0].get("noise_sigma", "") if seed_rows else ""
    method = seed_rows[0].get("perturbation_method", "") if seed_rows else ""

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Latent Perturbation Seed Analysis\n\n")
        f.write("This analysis groups perturbation results by seed only. It does not use successful/failed labels. Seed rows and plots are ordered by PSNR when PSNR is available.\n\n")
        if sample_id != "":
            f.write(f"- sample_id: `{int(sample_id):04d}`\n")
        if prompt:
            f.write(f"- prompt: `{prompt}`\n")
        if method:
            f.write(f"- perturbation_method: `{method}`\n")
        if noise_sigma != "":
            f.write(f"- noise_sigma: `{noise_sigma}`\n")
        f.write("\n")
        f.write("## Overall S0\n\n")
        f.write("| n | mean | variance | std | min | q05 | q10 | q25 | median | q75 | q90 | q95 | max |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {int(overall['S0_n'])} | {overall['S0_mean']:.6g} | {overall['S0_variance']:.6g} | "
            f"{overall['S0_std']:.6g} | {overall['S0_min']:.6g} | {overall['S0_q05']:.6g} | "
            f"{overall['S0_q10']:.6g} | {overall['S0_q25']:.6g} | {overall['S0_median']:.6g} | "
            f"{overall['S0_q75']:.6g} | {overall['S0_q90']:.6g} | {overall['S0_q95']:.6g} | "
            f"{overall['S0_max']:.6g} |\n\n"
        )

        f.write("## Per Seed S0\n\n")
        f.write("| seed | PSNR | n | mean | variance | std | min | q05 | q10 | q25 | median | q75 | q90 | q95 | max |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(seed_rows, key=seed_sort_key):
            psnr_text = f"{float(row['psnr']):.6g}" if row.get("psnr", "") != "" else ""
            f.write(
                f"| {row['seed']} | {psnr_text} | {int(row['S0_n'])} | {row['S0_mean']:.6g} | "
                f"{row['S0_variance']:.6g} | {row['S0_std']:.6g} | {row['S0_min']:.6g} | "
                f"{row['S0_q05']:.6g} | {row['S0_q10']:.6g} | {row['S0_q25']:.6g} | "
                f"{row['S0_median']:.6g} | {row['S0_q75']:.6g} | {row['S0_q90']:.6g} | "
                f"{row['S0_q95']:.6g} | {row['S0_max']:.6g} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/latent_perturbation_seed_sensitive_top10",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/latent_perturbation_sensitive",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perturb_rows = load_perturb_rows(Path(args.input_root))
    seed_rows = build_seed_summary(perturb_rows)
    overall_rows = build_overall_summary(perturb_rows)

    write_csv(output_dir / "per_perturb_terminal_sensitivity.csv", perturb_rows)
    write_csv(output_dir / "per_seed_terminal_sensitivity_summary.csv", seed_rows)
    write_csv(output_dir / "overall_terminal_sensitivity_summary.csv", overall_rows)
    seed_rows = sorted(seed_rows, key=seed_sort_key)
    plot_seed_box(perturb_rows, seed_rows, output_dir)
    plot_seed_summary_bars(seed_rows, output_dir)
    write_summary_md(output_dir, seed_rows, overall_rows)

    print(f"saved seed-grouped latent perturbation analysis to {output_dir}")


if __name__ == "__main__":
    main()
