import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


KEEP_COLUMNS = [
    "group",
    "pair_label",
    "seed_a",
    "seed_b",
    "alpha",
    "gen_P_sum",
    "gen_R_sum",
    "gen_rec_image_psnr",
    "rec_P_sum",
    "rec_R_sum",
    "first_div_t_ratio_gt_2",
]


DISPLAY_COLUMNS = [
    ("alpha", "alpha"),
    ("gen_P_sum", "gen P_t sum"),
    ("gen_R_sum", "gen R_t sum"),
    ("gen_rec_image_psnr", "rec PSNR"),
    ("rec_P_sum", "rec P_t sum"),
    ("rec_R_sum", "rec R_t sum"),
    ("first_div_t_ratio_gt_2", "first div. t"),
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def latex_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()
    return label or "table"


def display_group_name(text: str) -> str:
    return str(text).replace("_", "-")


def display_pair_label(text: str) -> str:
    return str(text).replace("_", "-")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: str, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def fmt_cell(key: str, value: str) -> str:
    if key == "first_div_t_ratio_gt_2":
        return value if value not in ("", None) else "--"
    digits = 2 if key == "gen_rec_image_psnr" else 4
    return fmt_float(value, digits)


def markdown_table(rows: List[Dict[str, str]], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(label for _, label in DISPLAY_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |")
    for row in rows:
        values = []
        for key, _ in DISPLAY_COLUMNS:
            values.append(fmt_cell(key, row[key]))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def latex_metric_cell(gen_value: str, rec_value: str) -> str:
    return rf"{fmt_float(gen_value)} / {fmt_float(rec_value)}"


def latex_table(rows: List[Dict[str, str]], title: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(title)}}}",
        rf"\label{{tab:{latex_label(label)}}}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$\alpha$ & P$_t$ sum (G/R) & R$_t$ sum (G/R) & PSNR & first div. $t$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    fmt_float(row["alpha"]),
                    latex_metric_cell(row["gen_P_sum"], row["rec_P_sum"]),
                    latex_metric_cell(row["gen_R_sum"], row["rec_R_sum"]),
                    fmt_float(row["gen_rec_image_psnr"], 2),
                    row["first_div_t_ratio_gt_2"] or "--",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def first_divergent_step(row: Dict[str, str], threshold: float) -> Dict[str, str]:
    image_path = row.get("reconstructed_image_path") or row.get("generated_image_path")
    if not image_path:
        return {"first_divergent_timestep": "", "first_divergent_loss": ""}

    trace_path = Path(image_path).parent / "trace_records.json"
    if not trace_path.exists():
        return {"first_divergent_timestep": "", "first_divergent_loss": ""}

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    losses = trace.get("convergence_losses", [])
    records = trace.get("inversion_records", [])
    for idx, loss in enumerate(losses):
        if float(loss) > threshold:
            record = records[idx] if idx < len(records) else {}
            return {
                "first_divergent_timestep": str(record.get("timestep", "")),
                "first_divergent_loss": str(loss),
            }
    return {"first_divergent_timestep": "", "first_divergent_loss": ""}


def first_ratio_divergent_step(row: Dict[str, str], ratio_threshold: float = 2.0) -> str:
    image_path = row.get("reconstructed_image_path") or row.get("generated_image_path")
    if not image_path:
        return ""

    trace_path = Path(image_path).parent / "trace_records.json"
    if not trace_path.exists():
        return ""

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    losses = [float(value) for value in trace.get("convergence_losses", [])]
    records = trace.get("inversion_records", [])
    for idx in range(1, len(losses)):
        if idx >= len(records):
            continue
        timestep = int(records[idx].get("timestep", -1))
        if timestep == 21:
            continue
        prev_loss = losses[idx - 1]
        if prev_loss <= 0:
            continue
        if losses[idx] / prev_loss > ratio_threshold:
            return str(timestep)
    return ""


def make_table_rows(group: str, rows: List[Dict[str, str]], divergence_loss_threshold: float) -> List[Dict[str, str]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["alpha"]))
    table_rows = []
    for row in sorted_rows:
        table_rows.append(
            {
                "group": group,
                "pair_label": row["pair_label"],
                "seed_a": row["seed_a"],
                "seed_b": row["seed_b"],
                "alpha": row["alpha"],
                "gen_P_sum": row["gen_P_sum"],
                "gen_R_sum": row["gen_R_sum"],
                "gen_rec_image_psnr": row["gen_rec_image_psnr"],
                "rec_P_sum": row["rec_P_sum"],
                "rec_R_sum": row["rec_R_sum"],
                "first_div_t_ratio_gt_2": first_ratio_divergent_step(row, ratio_threshold=2.0),
            }
        )
    return table_rows


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_root.glob("*/per_alpha_slerp_fpi_metrics.csv"))
    if not input_files:
        raise FileNotFoundError(f"No per_alpha_slerp_fpi_metrics.csv files found under {input_root}")

    combined_rows: List[Dict[str, str]] = []
    md_sections = []
    for csv_path in input_files:
        group = csv_path.parent.name
        rows = read_csv(csv_path)
        table_rows = make_table_rows(group, rows, args.divergence_loss_threshold)
        combined_rows.extend(table_rows)

        pair_label = table_rows[0]["pair_label"] if table_rows else ""
        seed_a = table_rows[0]["seed_a"] if table_rows else ""
        seed_b = table_rows[0]["seed_b"] if table_rows else ""
        title = (
            f"{display_group_name(group)} "
            f"({display_pair_label(pair_label)}, seed {seed_a} to seed {seed_b})"
        )

        write_csv(output_dir / f"{group}_metrics_by_alpha.csv", table_rows, KEEP_COLUMNS)
        md_text = markdown_table(table_rows, title)
        (output_dir / f"{group}_metrics_by_alpha.md").write_text(md_text, encoding="utf-8")
        tex_text = latex_table(table_rows, title, f"slerp_{group}_metrics_by_alpha")
        (output_dir / f"{group}_metrics_by_alpha.tex").write_text(tex_text, encoding="utf-8")
        md_sections.append(md_text)

    write_csv(output_dir / "combined_metrics_by_alpha.csv", combined_rows, KEEP_COLUMNS)
    (output_dir / "combined_metrics_by_alpha.md").write_text(
        "\n".join(md_sections),
        encoding="utf-8",
    )
    combined_tex_sections = []
    for csv_path in input_files:
        group = csv_path.parent.name
        group_rows = [row for row in combined_rows if row["group"] == group]
        if not group_rows:
            continue
        pair_label = group_rows[0]["pair_label"]
        seed_a = group_rows[0]["seed_a"]
        seed_b = group_rows[0]["seed_b"]
        title = (
            f"{display_group_name(group)} "
            f"({display_pair_label(pair_label)}, seed {seed_a} to seed {seed_b})"
        )
        combined_tex_sections.append(
            latex_table(group_rows, title, f"slerp_{group}_metrics_by_alpha")
        )
    (output_dir / "combined_metrics_by_alpha.tex").write_text(
        "\n".join(combined_tex_sections),
        encoding="utf-8",
    )
    print(f"saved tables to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact alpha-vs-metric tables from SLERP FPI per-alpha CSVs."
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
    parser.add_argument(
        "--divergence_loss_threshold",
        type=float,
        default=1e-6,
        help="First inversion timestep with FPI convergence loss above this value is marked divergent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
