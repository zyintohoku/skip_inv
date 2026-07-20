import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


METHOD_COLUMNS = [
    ("loss_gt_1e-6_t", r"loss $>10^{-6}$"),
    ("loss_gt_1e-9_t", r"loss $>10^{-9}$"),
    ("ratio_gt_2_t", r"ratio $>2$"),
    ("ratio_gt_3_t", r"ratio $>3$"),
]


CSV_COLUMNS = [
    "group",
    "pair_label",
    "seed_a",
    "seed_b",
    "alpha",
    "loss_gt_1e-6_t",
    "loss_gt_1e-6_loss",
    "loss_gt_1e-9_t",
    "loss_gt_1e-9_loss",
    "ratio_gt_2_t",
    "ratio_gt_2_ratio",
    "ratio_gt_2_prev_loss",
    "ratio_gt_2_loss",
    "ratio_gt_3_t",
    "ratio_gt_3_ratio",
    "ratio_gt_3_prev_loss",
    "ratio_gt_3_loss",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_trace_for_row(row: Dict[str, str]) -> Dict:
    image_path = row.get("reconstructed_image_path") or row.get("generated_image_path")
    if not image_path:
        return {}
    trace_path = Path(image_path).parent / "trace_records.json"
    if not trace_path.exists():
        return {}
    with trace_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def timestep_at(records: List[Dict], idx: int) -> str:
    if idx < len(records):
        return str(records[idx].get("timestep", ""))
    return ""


def record_timestep(records: List[Dict], idx: int) -> Optional[int]:
    if idx >= len(records):
        return None
    value = records[idx].get("timestep", "")
    if value == "":
        return None
    return int(value)


def first_loss_threshold(
    losses: List[float],
    records: List[Dict],
    threshold: float,
    min_timestep: Optional[int] = None,
) -> Dict[str, str]:
    for idx, loss in enumerate(losses):
        timestep = record_timestep(records, idx)
        if min_timestep is not None and (timestep is None or timestep < min_timestep):
            continue
        if loss > threshold:
            return {"t": timestep_at(records, idx), "loss": str(loss)}
    return {"t": "", "loss": ""}


def first_ratio_threshold(
    losses: List[float],
    records: List[Dict],
    threshold: float,
    skip_timesteps: Optional[List[int]] = None,
) -> Dict[str, str]:
    skip_set = set(skip_timesteps or [])
    for idx in range(1, len(losses)):
        timestep = record_timestep(records, idx)
        if timestep in skip_set:
            continue
        prev_loss = losses[idx - 1]
        loss = losses[idx]
        if prev_loss <= 0:
            continue
        ratio = loss / prev_loss
        if ratio > threshold:
            return {
                "t": timestep_at(records, idx),
                "ratio": str(ratio),
                "prev_loss": str(prev_loss),
                "loss": str(loss),
            }
    return {"t": "", "ratio": "", "prev_loss": "", "loss": ""}


def divergence_row(group: str, row: Dict[str, str]) -> Dict[str, str]:
    trace = load_trace_for_row(row)
    losses = [float(value) for value in trace.get("convergence_losses", [])]
    records = trace.get("inversion_records", [])

    loss_1e6 = first_loss_threshold(losses, records, 1e-6, min_timestep=401)
    loss_1e9 = first_loss_threshold(losses, records, 1e-9, min_timestep=401)
    ratio_2 = first_ratio_threshold(losses, records, 2.0, skip_timesteps=[21])
    ratio_3 = first_ratio_threshold(losses, records, 3.0, skip_timesteps=[21])

    return {
        "group": group,
        "pair_label": row["pair_label"],
        "seed_a": row["seed_a"],
        "seed_b": row["seed_b"],
        "alpha": row["alpha"],
        "loss_gt_1e-6_t": loss_1e6["t"],
        "loss_gt_1e-6_loss": loss_1e6["loss"],
        "loss_gt_1e-9_t": loss_1e9["t"],
        "loss_gt_1e-9_loss": loss_1e9["loss"],
        "ratio_gt_2_t": ratio_2["t"],
        "ratio_gt_2_ratio": ratio_2["ratio"],
        "ratio_gt_2_prev_loss": ratio_2["prev_loss"],
        "ratio_gt_2_loss": ratio_2["loss"],
        "ratio_gt_3_t": ratio_3["t"],
        "ratio_gt_3_ratio": ratio_3["ratio"],
        "ratio_gt_3_prev_loss": ratio_3["prev_loss"],
        "ratio_gt_3_loss": ratio_3["loss"],
    }


def fmt_alpha(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def fmt_timestep(value: str) -> str:
    return value if value else "--"


def latex_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return label or "table"


def display_name(text: str) -> str:
    return str(text).replace("_", "-")


def latex_table(rows: List[Dict[str, str]], title: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{title}}}",
        rf"\label{{tab:{latex_label(label)}}}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\alpha$ & " + " & ".join(header for _, header in METHOD_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [fmt_alpha(row["alpha"])]
                + [fmt_timestep(row[column]) for column, _ in METHOD_COLUMNS]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_root.glob("*/per_alpha_slerp_fpi_metrics.csv"))
    if not input_files:
        raise FileNotFoundError(f"No per_alpha_slerp_fpi_metrics.csv files found under {input_root}")

    combined_rows: List[Dict[str, str]] = []
    combined_tex_sections = []
    for csv_path in input_files:
        group = csv_path.parent.name
        rows = sorted(read_csv(csv_path), key=lambda row: float(row["alpha"]))
        table_rows = [divergence_row(group, row) for row in rows]
        combined_rows.extend(table_rows)
        write_csv(output_dir / f"{group}_divergence_by_alpha.csv", table_rows)

        pair_label = table_rows[0]["pair_label"] if table_rows else ""
        seed_a = table_rows[0]["seed_a"] if table_rows else ""
        seed_b = table_rows[0]["seed_b"] if table_rows else ""
        title = f"{display_name(group)} ({display_name(pair_label)}, seed {seed_a} to seed {seed_b})"
        tex = latex_table(table_rows, title, f"slerp_{group}_divergence_by_alpha")
        (output_dir / f"{group}_divergence_by_alpha.tex").write_text(tex, encoding="utf-8")
        combined_tex_sections.append(tex)

    write_csv(output_dir / "combined_divergence_by_alpha.csv", combined_rows)
    (output_dir / "combined_divergence_by_alpha.tex").write_text(
        "\n".join(combined_tex_sections),
        encoding="utf-8",
    )
    print(f"saved divergence tables to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create divergence-timestep comparison tables from SLERP FPI traces."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/slerp_fpi_gs7_seed_sensitive_sample0443",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/slerp_fpi_gs7_seed_sensitive_sample0443_metric_tables",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
