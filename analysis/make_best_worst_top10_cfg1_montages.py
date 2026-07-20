#!/usr/bin/env python3
"""Make 10x10 best/worst CFG=1 generation montages."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path("results/fpi_gs7_seed_psnr/best_worst_top10_cfg1_init_latent_generation")


def parse_int_set(text: str) -> list[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(chunk))
    return values


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    image.thumbnail((size, size), resample)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def read_selected_prompts(root: Path, label: str, top_k: int) -> list[dict]:
    path = root / "selected_prompts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected prompt CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["label"] == label]
    rows = sorted(rows, key=lambda row: int(row["rank"]))[:top_k]
    if len(rows) != top_k:
        raise ValueError(f"Expected {top_k} {label} rows in {path}, found {len(rows)}")
    return rows


def draw_overlay(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=(255, 255, 255))
    draw.text((x, y), text, font=font, fill=(15, 15, 15))


def make_montage(root: Path, label: str, seeds: list[int], args: argparse.Namespace) -> Path:
    rows = read_selected_prompts(root, label, args.top_k)
    cell = args.cell_size
    gap = args.gap
    cols = len(seeds)
    image_rows = len(rows)
    width = cols * cell + (cols - 1) * gap
    height = image_rows * cell + (image_rows - 1) * gap
    canvas = Image.new("RGB", (width, height), args.background)
    draw = ImageDraw.Draw(canvas)
    overlay_font = load_font(args.overlay_font_size)

    missing = []
    for row_idx, row in enumerate(rows):
        sample_id = int(row["sample_id"])
        sample_dir = root / label / f"sample_{sample_id:04d}"
        for col_idx, seed in enumerate(seeds):
            path = sample_dir / f"seed_{seed:02d}_cfg{args.guidance_scale:g}_gen.png"
            x = col_idx * (cell + gap)
            y = row_idx * (cell + gap)
            if not path.exists():
                missing.append(path)
                tile = Image.new("RGB", (cell, cell), (245, 245, 245))
            else:
                tile = fit_image(path, cell)
            canvas.paste(tile, (x, y))
            if args.overlay:
                draw_overlay(draw, f"r{row_idx + 1} s{seed}", (x + 6, y + 5), overlay_font)

    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} images while making {label} montage:\n{preview}")

    output_path = root / f"{label}_10x10_cfg{args.guidance_scale:g}_gen.png"
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create best/worst 10x10 CFG generation montages.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--cell_size", type=int, default=160)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--background", default="white")
    parser.add_argument("--overlay", action="store_true", help="Draw rank/seed tags inside each tile.")
    parser.add_argument("--overlay_font_size", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    seeds = parse_int_set(args.seeds)
    outputs = [make_montage(root, label, seeds, args) for label in ("best", "worst")]
    for path in outputs:
        print(f"saved montage: {path}")


if __name__ == "__main__":
    main()
