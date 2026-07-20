import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch


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


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_trace_rows(trace_root: Path, label: str, sample_id: int):
    sample_dir = trace_root / label / f"sample_{sample_id:04d}"
    rows = []
    for json_path in sorted(sample_dir.glob("seed_*/prompt_pressure_trace.json")):
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        seed = int(data["seed"])
        for record in data["records"]:
            rows.append(
                {
                    "label": label,
                    "sample_id": sample_id,
                    "seed": seed,
                    "trace_json_path": str(json_path),
                    **record,
                }
            )
    return rows


def latent_step_lengths(latent_trace: torch.Tensor):
    diffs = latent_trace[1:].float() - latent_trace[:-1].float()
    return torch.linalg.vector_norm(diffs.flatten(start_dim=1), dim=1).cpu().tolist()


def plot_raw_pressure(records, output_path: Path, sample_id: int, seed: int, prompt: str):
    steps = [r["step_index"] for r in records]
    p_values = [r["prompt_pressure_P_t"] for r in records]
    r_values = [r["relative_pressure_R_t"] for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, p_values, color="tab:blue", linewidth=1.8)
    axes[0].set_ylabel("raw P_t")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, r_values, color="tab:orange", linewidth=1.8)
    axes[1].set_ylabel("raw R_t")
    axes[1].set_xlabel("DDIM step index")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"sample {sample_id}, seed {seed}: raw P_t / R_t\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latent_step_lengths(trace_tensors_path: Path, output_path: Path, sample_id: int, seed: int, prompt: str):
    tensors = torch.load(trace_tensors_path, map_location="cpu")
    text_lengths = latent_step_lengths(tensors["text_latent_trace"])
    uncond_lengths = latent_step_lengths(tensors["uncond_latent_trace"])
    steps = list(range(len(text_lengths)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, text_lengths, color="tab:blue", linewidth=1.8)
    axes[0].set_ylabel("CFG latent step length")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, uncond_lengths, color="tab:gray", linewidth=1.8)
    axes[1].set_ylabel("Uncond latent step length")
    axes[1].set_xlabel("DDIM step index")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"sample {sample_id}, seed {seed}: latent step lengths\n{prompt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_per_seed_plots(expanded_rows):
    pressure_dir = expanded_rows[0]["output_dir"] / "plots_raw_pressure_by_seed"
    step_dir = expanded_rows[0]["output_dir"] / "plots_latent_step_lengths_by_seed"

    for row in expanded_rows:
        seed = int(row["seed"])
        trace_json_path = Path(row["trace_json_path"])
        with trace_json_path.open(encoding="utf-8") as f:
            trace = json.load(f)
        plot_raw_pressure(
            trace["records"],
            pressure_dir / f"sample_{int(row['sample_id']):04d}_seed_{seed:06d}_raw_P_R.png",
            int(row["sample_id"]),
            seed,
            row["prompt"],
        )
        plot_latent_step_lengths(
            Path(row["trace_tensors_path"]),
            step_dir / f"sample_{int(row['sample_id']):04d}_seed_{seed:06d}_latent_step_lengths.png",
            int(row["sample_id"]),
            seed,
            row["prompt"],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Expand one seed-sensitive sample across seeds and join reconstruction PSNR/image paths."
    )
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--label", type=str, default="seed_sensitive")
    parser.add_argument("--trace_root", type=str, default="outputs/prompt_pressure_seed_sensitive_top10")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="results/prompt_pressure_seed_sensitive_top10_analysis/distribution_metrics",
    )
    parser.add_argument(
        "--psnr_detail",
        type=str,
        default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/prompt_pressure_seed_sensitive_top10_analysis/sample_drilldown",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    trace_root = Path(args.trace_root)
    output_dir = Path(args.output_root) / f"sample_{args.sample_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics = [
        row
        for row in read_csv_rows(Path(args.metrics_dir) / "per_seed_distribution_metrics.csv")
        if row["label"] == args.label and int(row["sample_id"]) == args.sample_id
    ]
    if not per_seed_metrics:
        raise FileNotFoundError(
            f"No per-seed metrics found for label={args.label} sample_id={args.sample_id}"
        )

    psnr_rows = [
        row
        for row in read_csv_rows(Path(args.psnr_detail))
        if int(row["sample_id"]) == args.sample_id
    ]
    psnr_by_seed = {int(row["seed"]): row for row in psnr_rows}

    expanded_rows = []
    for metric_row in sorted(per_seed_metrics, key=lambda r: int(r["seed"])):
        seed = int(metric_row["seed"])
        psnr_row = psnr_by_seed.get(seed, {})
        seed_dir = trace_root / args.label / f"sample_{args.sample_id:04d}" / f"seed_{seed:06d}"
        row = {
            **metric_row,
            "psnr": psnr_row.get("psnr", ""),
            "mse": psnr_row.get("mse", ""),
            "gen_path": psnr_row.get("gen_path", ""),
            "rec_path": psnr_row.get("rec_path", ""),
            "prompt_pressure_generated_path": str(seed_dir / "generated.png"),
            "prompt_pressure_uncond_generated_path": str(seed_dir / "uncond_generated.png"),
            "trace_json_path": str(seed_dir / "prompt_pressure_trace.json"),
            "trace_tensors_path": str(seed_dir / "trace_tensors.pt"),
        }
        expanded_rows.append(row)

    write_csv(output_dir / "per_seed_expanded_with_psnr.csv", expanded_rows)
    write_per_seed_plots([{**row, "output_dir": output_dir} for row in expanded_rows])

    trace_rows = load_trace_rows(trace_root, args.label, args.sample_id)
    step_rows = []
    for row in trace_rows:
        seed = int(row["seed"])
        psnr_row = psnr_by_seed.get(seed, {})
        step_rows.append(
            {
                **row,
                "psnr": psnr_row.get("psnr", ""),
                "mse": psnr_row.get("mse", ""),
                "gen_path": psnr_row.get("gen_path", ""),
                "rec_path": psnr_row.get("rec_path", ""),
            }
        )
    write_csv(output_dir / "per_step_prompt_pressure_with_psnr.csv", step_rows)

    prompt = expanded_rows[0]["prompt"]
    mapping_key = expanded_rows[0]["mapping_key"]
    psnr_values = [float(r["psnr"]) for r in expanded_rows if r["psnr"]]
    p_sums = [float(r["P_raw_sum"]) for r in expanded_rows]
    r_sums = [float(r["R_raw_sum"]) for r in expanded_rows]
    delta_sums = [float(r["Delta_raw_sum"]) for r in expanded_rows]

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(f"# Seed-Sensitive Sample {args.sample_id}\n\n")
        f.write(f"- label: `{args.label}`\n")
        f.write(f"- mapping_key: `{mapping_key}`\n")
        f.write(f"- prompt: {prompt}\n")
        f.write(f"- seeds: {len(expanded_rows)}\n")
        if psnr_values:
            f.write(
                f"- PSNR: mean={sum(psnr_values)/len(psnr_values):.4f}, "
                f"min={min(psnr_values):.4f}, max={max(psnr_values):.4f}\n"
            )
        f.write(f"- P_raw_sum mean: {sum(p_sums)/len(p_sums):.4f}\n")
        f.write(f"- R_raw_sum mean: {sum(r_sums)/len(r_sums):.4f}\n")
        f.write(f"- Delta_raw_sum mean: {sum(delta_sums)/len(delta_sums):.4f}\n\n")

        f.write("## Per Seed\n\n")
        f.write(
            "| seed | PSNR | MSE | P sum | P late | R sum | Delta sum | gen | rec | trace |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n")
        for row in sorted(expanded_rows, key=lambda r: int(r["seed"])):
            gen_path = relpath(Path(row["gen_path"]), repo_root) if row["gen_path"] else ""
            rec_path = relpath(Path(row["rec_path"]), repo_root) if row["rec_path"] else ""
            trace_path = relpath(Path(row["trace_json_path"]), repo_root)
            f.write(
                f"| {row['seed']} | {float(row['psnr']):.4f} | {float(row['mse']):.4f} | "
                f"{float(row['P_raw_sum']):.4f} | {float(row['P_norm_late_30_49']):.4f} | "
                f"{float(row['R_raw_sum']):.4f} | {float(row['Delta_raw_sum']):.4f} | "
                f"`{gen_path}` | `{rec_path}` | `{trace_path}` |\n"
            )

    print(f"saved sample drilldown to {output_dir}")
    print(f"per-seed csv: {output_dir / 'per_seed_expanded_with_psnr.csv'}")
    print(f"per-step csv: {output_dir / 'per_step_prompt_pressure_with_psnr.csv'}")
    print(f"summary: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
