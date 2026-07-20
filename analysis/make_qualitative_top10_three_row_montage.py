#!/usr/bin/env python3
"""Build paper-style 3-row qualitative gen/rec montages."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUAL_ROOT = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "qualitative_samples_top10"
OUT_DIR = QUAL_ROOT / "selected_gen_rec_pairs" / "top10_3x6"
CUSTOM_OUT_DIR = QUAL_ROOT / "selected_gen_rec_pairs" / "top10_custom_74_56_443"
PROMPT_CSV = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "fpi_gs7_seed_psnr_by_sample.csv"

IMAGE_SIZE = 512
INTERNAL_GAP = 16
PAIR_GAP = 48
LABEL_H = 170
ROW_W = 6480
ROW_H = LABEL_H + IMAGE_SIZE


def font(size: int) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


LABEL_FONT = font(85)


def read_prompt(sample_id: int) -> str:
    with PROMPT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["sample_id"]) == sample_id:
                return row["original_prompt"]
    raise KeyError(f"sample_id {sample_id} not found in {PROMPT_CSV}")


def make_label(title: str, sample_id: int) -> Image.Image:
    text = f"{title}: {read_prompt(sample_id)}"
    label = Image.new("RGB", (ROW_W, LABEL_H), "white")
    draw = ImageDraw.Draw(label)
    box = draw.textbbox((0, 0), text, font=LABEL_FONT)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    draw.text(
        ((ROW_W - text_w) // 2, (LABEL_H - text_h) // 2 - box[1]),
        text,
        font=LABEL_FONT,
        fill=(20, 20, 20),
    )
    return label


def make_pair(sample_id: int, seed: int) -> Image.Image:
    sample_dir = QUAL_ROOT / f"sample_{sample_id:04d}"
    gen = Image.open(sample_dir / f"seed_{seed:02d}_gen.png").convert("RGB")
    rec = Image.open(sample_dir / f"seed_{seed:02d}_rec.png").convert("RGB")
    pair = Image.new("RGB", (2 * IMAGE_SIZE + INTERNAL_GAP, IMAGE_SIZE), "white")
    pair.paste(gen, (0, 0))
    pair.paste(rec, (IMAGE_SIZE + INTERNAL_GAP, 0))
    return pair


def make_images_row(sample_id: int, seeds: list[int]) -> Image.Image:
    row = Image.new("RGB", (ROW_W, IMAGE_SIZE), "white")
    x = 0
    for seed in seeds:
        pair = make_pair(sample_id, seed)
        row.paste(pair, (x, 0))
        x += pair.width + PAIR_GAP
    return row


def combine_row(label: Image.Image, images: Image.Image) -> Image.Image:
    row = Image.new("RGB", (ROW_W, ROW_H), "white")
    row.paste(label.convert("RGB"), (0, 0))
    row.paste(images.convert("RGB"), (0, LABEL_H))
    return row


def main() -> None:
    CUSTOM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    easy_row = combine_row(
        make_label("Easy prompt", 308),
        make_images_row(308, [1, 2, 3, 4, 5, 6]),
    )
    hard_row = combine_row(
        make_label("Hard prompt", 56),
        make_images_row(56, [1, 6, 7, 8, 9, 10]),
    )
    intermediate_row = combine_row(
        make_label("Intermediate prompt", 443),
        make_images_row(443, [2, 6, 3, 10, 7, 4]),
    )

    montage = Image.new("RGB", (ROW_W, 3 * ROW_H), "white")
    for idx, row in enumerate([easy_row, hard_row, intermediate_row]):
        montage.paste(row, (0, idx * ROW_H))

    png_path = CUSTOM_OUT_DIR / "sample308_56_443_rows_big_prompt.png"
    pdf_path = CUSTOM_OUT_DIR / "sample308_56_443_rows_big_prompt.pdf"
    montage.save(png_path)
    montage.save(pdf_path, "PDF", resolution=300.0)

    temp_dir = PROJECT_ROOT / "results" / "temp_plot"
    temp_dir.mkdir(parents=True, exist_ok=True)
    montage.save(temp_dir / png_path.name)
    montage.save(temp_dir / pdf_path.name, "PDF", resolution=300.0)
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    print(f"copied to: {temp_dir}")


if __name__ == "__main__":
    main()
