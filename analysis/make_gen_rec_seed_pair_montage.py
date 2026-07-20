import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_prompt(mapping_file: Path, sample_id: int) -> str:
    with mapping_file.open(encoding="utf-8") as f:
        items = list(json.load(f).items())
    item = items[sample_id][1]
    return item["original_prompt"]


def load_psnr(psnr_csv: Path, sample_id: int):
    rows = {}
    with psnr_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["sample_id"]) == sample_id:
                rows[int(row["seed"])] = float(row["psnr"])
    return rows


def sample_ids_from_csv(path: Path, top_k: int):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return set()
    key = "sample_id" if "sample_id" in rows[0] else next(iter(rows[0]))
    return {int(row[key]) for row in rows[:top_k]}


def infer_sample_label(args, sample_id: int) -> str:
    if args.label != "auto":
        return args.label
    label_sources = [
        ("best", Path(args.best_csv)),
        ("worst", Path(args.worst_csv)),
        ("sensitive", Path(args.sensitive_csv)),
    ]
    for label, path in label_sources:
        if sample_id in sample_ids_from_csv(path, args.top_k):
            return label
    return "unknown"


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_centered_text(draw, box, text, text_font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=4, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.multiline_text((x, y), text, font=text_font, fill=fill, spacing=4, align="center")


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    image.thumbnail((size, size), resample)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_montage(args):
    sample_id = args.sample_id
    seeds = list(range(args.seed_start, args.seed_end + 1))
    prompt = load_prompt(Path(args.mapping_file), sample_id)
    psnr_by_seed = load_psnr(Path(args.psnr_csv), sample_id)
    sample_label = infer_sample_label(args, sample_id)

    cell = args.cell_size
    label_h = args.label_height
    title_h = args.title_height
    margin = args.margin
    gap = args.gap
    cols = args.cols
    rows = args.rows

    width = margin * 2 + cols * cell + (cols - 1) * gap
    height = margin * 2 + title_h + rows * (label_h + cell) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = font(args.title_font_size, bold=True)
    label_font = font(args.label_font_size, bold=True)

    prompt_clean = prompt.replace("[", "").replace("]", "")
    wrapped_prompt = "\n".join(
        textwrap.wrap(f"{sample_label} | sample {sample_id:04d}: {prompt_clean}", width=args.prompt_wrap)
    )
    draw_centered_text(draw, (margin, margin, width - margin, margin + title_h), wrapped_prompt, title_font)

    y0 = margin + title_h
    for pair_idx, seed in enumerate(seeds):
        row = pair_idx // 2
        pair_col = pair_idx % 2
        x_gen = margin + pair_col * 2 * (cell + gap)
        x_rec = x_gen + cell + gap
        y_label = y0 + row * (label_h + cell + gap)
        y_img = y_label + label_h

        gen_path = Path(args.outputs_pattern.format(seed=seed)) / f"{sample_id}gen.png"
        rec_path = Path(args.outputs_pattern.format(seed=seed)) / f"{sample_id}rec.png"
        if not gen_path.exists() or not rec_path.exists():
            raise FileNotFoundError(f"Missing gen/rec image for sample {sample_id}, seed {seed}")

        psnr = psnr_by_seed.get(seed)
        gen_label = f"seed {seed}\noriginal"
        rec_label = f"seed {seed}\nreconstruction"
        if psnr is not None:
            rec_label += f"\nPSNR {psnr:.2f}"

        draw_centered_text(draw, (x_gen, y_label, x_gen + cell, y_img), gen_label, label_font)
        draw_centered_text(draw, (x_rec, y_label, x_rec + cell, y_img), rec_label, label_font)

        canvas.paste(fit_image(gen_path, cell), (x_gen, y_img))
        canvas.paste(fit_image(rec_path, cell), (x_rec, y_img))

        draw.rectangle((x_gen, y_img, x_gen + cell - 1, y_img + cell - 1), outline=(210, 210, 210), width=1)
        draw.rectangle((x_rec, y_img, x_rec + cell - 1, y_img + cell - 1), outline=(210, 210, 210), width=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"saved montage: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--mapping_file", type=str, default="PIE_bench/mapping_file.json")
    parser.add_argument("--psnr_csv", type=str, default="results/aidi_gs7_seed_psnr/aidi_gs7_seed_psnr_detail.csv")
    parser.add_argument("--best_csv", type=str, default="results/aidi_gs7_seed_psnr/prompt_psnr_best30.csv")
    parser.add_argument("--worst_csv", type=str, default="results/aidi_gs7_seed_psnr/prompt_psnr_worst30.csv")
    parser.add_argument(
        "--sensitive_csv",
        type=str,
        default="results/aidi_gs7_seed_psnr/prompt_psnr_most_seed_sensitive30.csv",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--label", type=str, default="auto")
    parser.add_argument("--outputs_pattern", type=str, default="outputs/aidi_gs7_seed{seed}")
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=10)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cell_size", type=int, default=224)
    parser.add_argument("--label_height", type=int, default=64)
    parser.add_argument("--title_height", type=int, default=104)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--gap", type=int, default=14)
    parser.add_argument("--title_font_size", type=int, default=24)
    parser.add_argument("--label_font_size", type=int, default=16)
    parser.add_argument("--prompt_wrap", type=int, default=58)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    make_montage(parse_args())
