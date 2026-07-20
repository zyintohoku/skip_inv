#!/usr/bin/env python3
"""Build a best-top10 FPI qualitative montage with vertical gen/rec pairs."""

import argparse
import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUAL_ROOT = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "qualitative_samples_top10"
DEFAULT_OUT = DEFAULT_QUAL_ROOT / "best_top10_seed10_gen_rec_vertical.png"


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
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def load_best_rows(qual_root: Path, top_k: int) -> list[dict]:
    manifest = qual_root / "top10_prompt_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["group"] == "best"]
    rows = rows[:top_k]
    if len(rows) != top_k:
        raise ValueError(f"Expected {top_k} best rows in {manifest}, found {len(rows)}")
    return rows


def centered_text(draw: ImageDraw.ImageDraw, xywh: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y, w, h = xywh
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.multiline_text(
        (x + (w - text_w) // 2 - bbox[0], y + (h - text_h) // 2 - bbox[1]),
        text,
        font=font,
        fill=(20, 20, 20),
        spacing=4,
        align="center",
    )


def wrap_prompt(text: str, max_chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False))


def clean_prompt(text: str) -> str:
    return text.replace("[", "").replace("]", "")


def make_montage(args: argparse.Namespace) -> Path:
    qual_root = Path(args.qual_root)
    rows = load_best_rows(qual_root, args.top_k)
    seeds = parse_int_set(args.seeds)

    cell = args.cell_size
    pair_gap = args.pair_gap
    col_gap = args.col_gap
    row_gap = args.row_gap
    prompt_h = args.prompt_height
    seed_h = args.seed_header_height
    margin = args.margin

    cols = len(seeds)
    grid_w = cols * cell + (cols - 1) * col_gap
    pair_h = 2 * cell + pair_gap
    row_h = prompt_h + pair_h
    width = 2 * margin + grid_w
    height = 2 * margin + seed_h + len(rows) * row_h + (len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    prompt_font = load_font(args.prompt_font_size, bold=True)

    missing = []
    y = margin + seed_h
    for row in rows:
        sample_id = int(row["sample_id"])
        sample_dir = qual_root / f"sample_{sample_id:04d}"
        prompt = wrap_prompt(clean_prompt(row["editing_prompt"]), args.prompt_wrap_chars)
        centered_text(draw, (margin, y, grid_w, prompt_h), prompt, prompt_font)

        image_y = y + prompt_h
        for col_idx, seed in enumerate(seeds):
            x = margin + col_idx * (cell + col_gap)
            gen_path = sample_dir / f"seed_{seed:02d}_gen.png"
            rec_path = sample_dir / f"seed_{seed:02d}_rec.png"
            if not gen_path.exists():
                missing.append(gen_path)
                gen = Image.new("RGB", (cell, cell), (245, 245, 245))
            else:
                gen = fit_square(gen_path, cell)
            if not rec_path.exists():
                missing.append(rec_path)
                rec = Image.new("RGB", (cell, cell), (245, 245, 245))
            else:
                rec = fit_square(rec_path, cell)
            canvas.paste(gen, (x, image_y))
            canvas.paste(rec, (x, image_y + cell + pair_gap))
        y += row_h + row_gap

    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} images:\n{preview}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    if args.pdf:
        canvas.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual_root", default=str(DEFAULT_QUAL_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--cell_size", type=int, default=160)
    parser.add_argument("--col_gap", type=int, default=18)
    parser.add_argument("--pair_gap", type=int, default=6)
    parser.add_argument("--row_gap", type=int, default=28)
    parser.add_argument("--prompt_height", type=int, default=58)
    parser.add_argument("--seed_header_height", type=int, default=0)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--prompt_font_size", type=int, default=24)
    parser.add_argument("--prompt_wrap_chars", type=int, default=92)
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    out_path = make_montage(parse_args())
    print(f"saved: {out_path}")
    if out_path.with_suffix(".pdf").exists():
        print(f"saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
