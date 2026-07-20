import argparse
import csv
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/cfg_directional_features_seed_sensitive_top10/seed_sensitive",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/cfg_directional_features_seed_sensitive_top10/per_seed_cfg_directional_features.csv",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rows = []
    for path in sorted(input_root.glob("sample_*/per_seed_cfg_directional_features.csv")):
        rows.extend(read_rows(path))
    if not rows:
        raise FileNotFoundError(f"No sample summaries found under {input_root}")
    write_csv(Path(args.output_csv), rows)
    print(f"saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
