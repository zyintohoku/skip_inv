import argparse
import csv
import json
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


STAGES = [
    ("early_0_9", 0, 10),
    ("mid_10_29", 10, 30),
    ("late_30_49", 30, 50),
]


def load_psnr_detail(path: Path):
    values = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values[(int(row["sample_id"]), int(row["seed"]))] = float(row["psnr"])
    return values


def dot(a, b):
    return torch.dot(a.detach().float().reshape(-1), b.detach().float().reshape(-1))


def norm(a):
    return torch.linalg.vector_norm(a.detach().float().reshape(-1))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def read_trace_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compute_trace_rows(trace_json_path: Path, psnr_by_key):
    trace = read_trace_json(trace_json_path)
    tensor_path = trace_json_path.parent / "trace_tensors.pt"
    tensors = torch.load(tensor_path, map_location="cpu")
    noise_uncond = tensors["noise_pred_uncond"]
    noise_text = tensors["noise_pred_text"]

    sample_id = int(trace["sample_id"])
    seed = int(trace["seed"])
    guidance_scale = float(trace.get("guidance_scale", 1.0))
    psnr = psnr_by_key.get((sample_id, seed), float("nan"))
    prompt = trace["prompt"]

    rows = []
    for record, eps_u, eps_c in zip(trace["records"], noise_uncond, noise_text):
        delta = eps_c - eps_u
        scaled_delta = guidance_scale * delta
        eps_u_norm = norm(eps_u)
        scaled_delta_norm = norm(scaled_delta)
        delta_norm = norm(delta)
        product = eps_u_norm * scaled_delta_norm
        eps_dot_guidance = dot(eps_u, scaled_delta)
        cos_value = eps_dot_guidance / product.clamp_min(1e-12)
        distance = norm(eps_u - scaled_delta)

        rows.append(
            {
                "label": trace.get("label", "seed_sensitive"),
                "sample_id": sample_id,
                "seed": seed,
                "prompt": prompt,
                "psnr": psnr,
                "step_index": int(record["step_index"]),
                "timestep": int(record["timestep"]),
                "guidance_scale": guidance_scale,
                "epsilon_uncond_l2": float(eps_u_norm.item()),
                "guidance_delta_l2": float(delta_norm.item()),
                "scaled_guidance_delta_l2": float(scaled_delta_norm.item()),
                "epsilon_uncond_dot_scaled_guidance": float(eps_dot_guidance.item()),
                "epsilon_uncond_scaled_guidance_cos": float(cos_value.item()),
                "epsilon_uncond_minus_scaled_guidance_l2": float(distance.item()),
            }
        )
    return rows


def summarize_seed(rows):
    cos_values = [row["epsilon_uncond_scaled_guidance_cos"] for row in rows]
    summary = {
        "label": rows[0]["label"],
        "sample_id": rows[0]["sample_id"],
        "seed": rows[0]["seed"],
        "prompt": rows[0]["prompt"],
        "psnr": rows[0]["psnr"],
        "cos_mean": mean(cos_values),
        "cos_std": stdev(cos_values),
        "cos_min": min(cos_values),
        "cos_max": max(cos_values),
        "cos_final_step": cos_values[-1],
        "distance_mean": mean([row["epsilon_uncond_minus_scaled_guidance_l2"] for row in rows]),
    }
    for name, start, end in STAGES:
        stage_values = cos_values[start:end]
        summary[f"{name}_cos_mean"] = mean(stage_values)
        summary[f"{name}_cos_std"] = stdev(stage_values)
    return summary


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sample_curves(step_rows, output_dir: Path):
    grouped = defaultdict(list)
    psnr_values = []
    for row in step_rows:
        grouped[(row["sample_id"], row["seed"])].append(row)
        if math.isfinite(row["psnr"]):
            psnr_values.append(row["psnr"])
    norm_obj = Normalize(vmin=min(psnr_values), vmax=max(psnr_values))
    cmap = plt.get_cmap("viridis")

    by_sample = defaultdict(dict)
    for (sample_id, seed), rows in grouped.items():
        by_sample[sample_id][seed] = rows

    sample_dir = output_dir / "by_sample"
    for sample_id, seed_rows in sorted(by_sample.items()):
        prompt = next(iter(seed_rows.values()))[0]["prompt"]
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for seed, rows in sorted(seed_rows.items()):
            rows = sorted(rows, key=lambda row: row["step_index"])
            psnr = rows[0]["psnr"]
            ax.plot(
                [row["timestep"] for row in rows],
                [row["epsilon_uncond_scaled_guidance_cos"] for row in rows],
                color=cmap(norm_obj(psnr)),
                linewidth=1.8,
                alpha=0.9,
                label=f"seed {seed} | {psnr:.2f}",
            )
        ax.invert_xaxis()
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlabel("DDIM timestep")
        ax.set_ylabel(r"$\cos(\epsilon_t^u, s\delta_t)$")
        ax.set_title(
            f"sensitive | sample {sample_id:04d}\n{textwrap.shorten(prompt, width=88, placeholder='...')}",
            fontsize=11,
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
        sm = ScalarMappable(norm=norm_obj, cmap=cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.82)
        cbar_ax = fig.add_axes([0.86, 0.17, 0.025, 0.68])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Reconstruction PSNR")
        sample_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(sample_dir / f"sample_{sample_id:04d}_epsilon_guidance_cos_by_psnr.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_examples(seed_summary_rows, step_rows, output_dir: Path, sample_ids):
    step_by_key = defaultdict(list)
    for row in step_rows:
        step_by_key[(row["sample_id"], row["seed"])].append(row)

    by_sample = defaultdict(list)
    for row in seed_summary_rows:
        by_sample[row["sample_id"]].append(row)

    selected = []
    for sample_id in sample_ids:
        rows = sorted(by_sample[sample_id], key=lambda row: row["psnr"])
        selected.append(rows[0])
        selected.append(rows[-1])

    fig, axes = plt.subplots(len(selected), 1, figsize=(8.4, 2.55 * len(selected)), sharex=True, sharey=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, row in zip(axes, selected):
        rows = sorted(step_by_key[(row["sample_id"], row["seed"])], key=lambda item: item["step_index"])
        color = "tab:green" if row["psnr"] >= 50 else "tab:red"
        ax.plot(
            [item["timestep"] for item in rows],
            [item["epsilon_uncond_scaled_guidance_cos"] for item in rows],
            color=color,
            linewidth=1.8,
        )
        ax.invert_xaxis()
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        ax.grid(alpha=0.25)
        ax.set_ylabel(r"$\cos(\epsilon_t^u, s\delta_t)$")
        title = (
            f"sample {row['sample_id']:04d}, seed {row['seed']}, PSNR {row['psnr']:.2f}\n"
            f"{textwrap.shorten(row['prompt'], width=105, placeholder='...')}"
        )
        ax.set_title(title, fontsize=9.5, pad=9)
    axes[-1].set_xlabel("DDIM timestep")
    fig.suptitle(r"Examples of $\cos(\epsilon_t^u, s\delta_t)$ over time", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "epsilon_guidance_cos_time_examples_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_csv(output_dir / "selected_examples.csv", selected)


def write_correlation_summary(seed_summary_rows, output_path: Path):
    features = [
        "cos_mean",
        "early_0_9_cos_mean",
        "mid_10_29_cos_mean",
        "late_30_49_cos_mean",
        "cos_final_step",
        "distance_mean",
    ]
    psnr = [row["psnr"] for row in seed_summary_rows]
    rows = []
    for feature in features:
        xs = [row[feature] for row in seed_summary_rows]
        rows.append(
            {
                "feature": feature,
                "pearson_with_psnr": pearson(xs, psnr),
                "spearman_with_psnr": pearson(rank(xs), rank(psnr)),
                "mean": mean(xs),
                "std": stdev(xs),
                "min": min(xs),
                "max": max(xs),
            }
        )
    write_csv(output_path, rows)


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
        "--example_sample_ids",
        type=str,
        default="71,147,218,535",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/prompt_pressure_saved_latent_top10_merged_analysis/epsilon_guidance_cos_sensitive",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    psnr_by_key = load_psnr_detail(Path(args.psnr_detail_csv))

    all_step_rows = []
    seed_summary_rows = []
    trace_paths = sorted(input_root.glob("sample_*/seed_*/prompt_pressure_trace.json"))
    for trace_path in trace_paths:
        rows = compute_trace_rows(trace_path, psnr_by_key)
        all_step_rows.extend(rows)
        seed_summary_rows.append(summarize_seed(rows))

    write_csv(output_dir / "per_step_epsilon_guidance_cos.csv", all_step_rows)
    write_csv(output_dir / "per_seed_epsilon_guidance_cos_summary.csv", seed_summary_rows)
    write_correlation_summary(seed_summary_rows, output_dir / "epsilon_guidance_cos_psnr_correlations.csv")
    plot_sample_curves(all_step_rows, output_dir)
    example_sample_ids = [int(token.strip()) for token in args.example_sample_ids.split(",") if token.strip()]
    plot_examples(seed_summary_rows, all_step_rows, output_dir / "examples", example_sample_ids)

    print(f"saved {len(all_step_rows)} per-step rows")
    print(f"saved {len(seed_summary_rows)} per-seed rows")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
