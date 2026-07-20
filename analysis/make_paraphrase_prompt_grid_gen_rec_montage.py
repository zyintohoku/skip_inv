#!/usr/bin/env python3
"""Build a 3x10 gen/rec montage for the paraphrase prompt-grid output."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "paraphrase_prompt_grid_fpi_gs7_seed1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "fpi_gs7_seed_psnr"
    / "paraphrase_prompt_grid"
    / "paraphrase_prompt_grid_fpi_gs7_seed1_gen_rec_montage.png"
)
ROW_TITLES = [
    "Original prompt: a cat is sitting on a wooden table",
    "Original prompt: a squirrel is sitting on top of a wooden fence",
    "Original prompt: The Great Wave off Kanagawa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paraphrase prompt-grid gen/rec montage.")
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--pair_gap", type=int, default=3)
    parser.add_argument("--row_gap", type=int, default=20)
    parser.add_argument("--title_height", type=int, default=38)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--title_font_size", type=int, default=22)
    parser.add_argument("--background", default="white")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_square(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    image.thumbnail((size, size), resample)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (28, 28, 28),
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2
    draw.text((x, y), text, font=font, fill=fill)


def read_metrics(input_dir: Path) -> dict[int, dict[str, str]]:
    metrics_path = input_dir / "per_prompt_fpi_metrics.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return {int(row["prompt_id"]): row for row in csv.DictReader(f)}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row", "col", "prompt_id", "row_title", "prompt", "gen_image_path", "rec_image_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_montage(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    metrics = read_metrics(input_dir)

    cols = 10
    rows = 3
    cell_w = args.image_size
    cell_h = args.image_size * 2 + args.pair_gap
    grid_w = cols * cell_w + (cols - 1) * args.gap
    band_h = args.title_height + cell_h
    width = args.margin * 2 + grid_w
    height = args.margin * 2 + rows * band_h + (rows - 1) * args.row_gap

    canvas = Image.new("RGB", (width, height), args.background)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(args.title_font_size, bold=True)
    stroke = (218, 218, 218)
    manifest_rows = []

    for row_idx, row_title in enumerate(ROW_TITLES):
        band_top = args.margin + row_idx * (band_h + args.row_gap)
        draw_centered(draw, (args.margin, band_top, width - args.margin, band_top + args.title_height), row_title, title_font)
        y = band_top + args.title_height

        for col_idx in range(cols):
            prompt_id = row_idx * cols + col_idx
            x = args.margin + col_idx * (cell_w + args.gap)
            gen_path = input_dir / f"{prompt_id}gen.png"
            rec_path = input_dir / f"{prompt_id}rec.png"
            if not gen_path.exists() or not rec_path.exists():
                raise FileNotFoundError(f"Missing image pair for prompt_id={prompt_id}: {gen_path}, {rec_path}")

            canvas.paste(fit_square(gen_path, args.image_size), (x, y))
            canvas.paste(fit_square(rec_path, args.image_size), (x, y + args.image_size + args.pair_gap))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=stroke, width=1)
            draw.line(
                (
                    x,
                    y + args.image_size + args.pair_gap // 2,
                    x + cell_w - 1,
                    y + args.image_size + args.pair_gap // 2,
                ),
                fill=stroke,
                width=1,
            )

            metric_row = metrics.get(prompt_id, {})
            manifest_rows.append(
                {
                    "row": row_idx + 1,
                    "col": col_idx + 1,
                    "prompt_id": prompt_id,
                    "row_title": row_title,
                    "prompt": metric_row.get("prompt", ""),
                    "gen_image_path": str(gen_path.relative_to(PROJECT_ROOT)),
                    "rec_image_path": str(rec_path.relative_to(PROJECT_ROOT)),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".csv")
    write_manifest(manifest_path, manifest_rows)
    return output_path


if __name__ == "__main__":
    print(make_montage(parse_args()))
