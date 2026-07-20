#!/usr/bin/env python3
"""Build a random-7 montage from an FPI top-30 prompt set."""

import argparse
import csv
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_CSV = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "prompt_psnr_best30.csv"
DEFAULT_DETAIL_CSV = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "fpi_gs7_seed_psnr_detail.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "qualitative_best30_random7"
DEFAULT_OUTPUT = DEFAULT_OUT_DIR / "best30_random7_seed10_gen_rec_vertical.png"


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


def clean_prompt(text: str) -> str:
    return text.replace("[", "").replace("]", "")


def wrap_prompt(text: str, max_chars: int) -> str:
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False))


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


def fit_square(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    image.thumbnail((size, size), resample)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def read_prompt_rows(path: Path, rank_label: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for rank, row in enumerate(rows, start=1):
        row["top30_rank"] = str(rank)
        row["rank_label"] = rank_label
    return rows


def read_detail_paths(path: Path, sample_ids: set[int], seeds: set[int]) -> dict[tuple[int, int], dict]:
    paths = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = int(row["sample_id"])
            seed = int(row["seed"])
            if sample_id in sample_ids and seed in seeds:
                paths[(sample_id, seed)] = row
    return paths


def write_manifest(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "selection_order",
        "rank_label",
        "top30_rank",
        "sample_id",
        "mapping_key",
        "original_prompt",
        "editing_prompt",
        "editing_instruction",
        "psnr_mean",
        "psnr_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {name: row.get(name, "") for name in fieldnames}
            out["selection_order"] = idx
            writer.writerow(out)


def make_montage(args: argparse.Namespace) -> Path:
    prompt_rows = read_prompt_rows(Path(args.prompt_csv), args.label)
    if args.num_prompts > len(prompt_rows):
        raise ValueError(f"Cannot select {args.num_prompts} prompts from only {len(prompt_rows)} rows")

    rng = random.Random(args.random_seed)
    selected = rng.sample(prompt_rows, args.num_prompts)
    seeds = parse_int_set(args.seeds)
    detail = read_detail_paths(
        Path(args.detail_csv),
        {int(row["sample_id"]) for row in selected},
        set(seeds),
    )

    cell = args.cell_size
    pair_gap = args.pair_gap
    col_gap = args.col_gap
    row_gap = args.row_gap
    prompt_h = args.prompt_height
    margin = args.margin

    cols = len(seeds)
    grid_w = cols * cell + (cols - 1) * col_gap
    pair_h = 2 * cell + pair_gap
    row_h = prompt_h + pair_h
    width = 2 * margin + grid_w
    height = 2 * margin + len(selected) * row_h + (len(selected) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    prompt_font = load_font(args.prompt_font_size, bold=True)

    missing = []
    y = margin
    for row in selected:
        sample_id = int(row["sample_id"])
        prompt = wrap_prompt(clean_prompt(row["editing_prompt"]), args.prompt_wrap_chars)
        centered_text(draw, (margin, y, grid_w, prompt_h), prompt, prompt_font)

        image_y = y + prompt_h
        for col_idx, seed in enumerate(seeds):
            x = margin + col_idx * (cell + col_gap)
            detail_row = detail.get((sample_id, seed))
            if not detail_row:
                missing.append(f"sample {sample_id} seed {seed}")
                gen = Image.new("RGB", (cell, cell), (245, 245, 245))
                rec = Image.new("RGB", (cell, cell), (245, 245, 245))
            else:
                gen_path = PROJECT_ROOT / detail_row["gen_path"]
                rec_path = PROJECT_ROOT / detail_row["rec_path"]
                if not gen_path.exists():
                    missing.append(str(gen_path))
                    gen = Image.new("RGB", (cell, cell), (245, 245, 245))
                else:
                    gen = fit_square(gen_path, cell)
                if not rec_path.exists():
                    missing.append(str(rec_path))
                    rec = Image.new("RGB", (cell, cell), (245, 245, 245))
                else:
                    rec = fit_square(rec_path, cell)
            canvas.paste(gen, (x, image_y))
            canvas.paste(rec, (x, image_y + cell + pair_gap))
        y += row_h + row_gap

    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} image/data entries:\n{preview}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    if args.pdf:
        canvas.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    write_manifest(selected, out_path.with_name(out_path.stem + "_manifest.csv"))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_csv", default=str(DEFAULT_PROMPT_CSV))
    parser.add_argument("--detail_csv", default=str(DEFAULT_DETAIL_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--label", default="best")
    parser.add_argument("--num_prompts", type=int, default=7)
    parser.add_argument("--random_seed", type=int, default=20260527)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--cell_size", type=int, default=160)
    parser.add_argument("--col_gap", type=int, default=18)
    parser.add_argument("--pair_gap", type=int, default=6)
    parser.add_argument("--row_gap", type=int, default=28)
    parser.add_argument("--prompt_height", type=int, default=58)
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
    print(f"saved: {out_path.with_name(out_path.stem + '_manifest.csv')}")


if __name__ == "__main__":
    main()
