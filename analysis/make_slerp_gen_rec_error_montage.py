import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import clip


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resize_image(path: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    return image.resize((size, size), resample)


def make_error_image(gen_image: Image.Image, rec_image: Image.Image) -> Image.Image:
    gen = np.asarray(gen_image, dtype=np.float32)
    rec = np.asarray(rec_image, dtype=np.float32)
    err = np.abs(gen - rec).mean(axis=2)
    max_err = float(err.max())
    if max_err > 0:
        err = err / max_err * 255.0
    err_u8 = np.clip(err, 0, 255).astype(np.uint8)

    # Red-on-black error map: spatial structure is clearer than raw grayscale at small sizes.
    rgb = np.zeros((*err_u8.shape, 3), dtype=np.uint8)
    rgb[..., 0] = err_u8
    rgb[..., 1] = (err_u8 * 0.35).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def load_loss_curve(row: Dict[str, str]) -> Tuple[List[int], List[float]]:
    image_path = row.get("reconstructed_image_path") or row.get("generated_image_path")
    if not image_path:
        raise ValueError("Row does not contain generated/reconstructed image paths.")
    trace_path = Path(image_path).parent / "trace_records.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace file: {trace_path}")

    with trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    records = trace.get("inversion_records", [])
    losses = trace.get("convergence_losses")
    if losses is None:
        losses = [record["convergence_loss"] for record in records]
    if len(records) != len(losses):
        raise ValueError(f"record/loss length mismatch in {trace_path}")

    paired = sorted(
        (
            int(record["timestep"]),
            -math.log10(max(float(loss), 1e-30)),
        )
        for record, loss in zip(records, losses)
    )
    return [item[0] for item in paired], [item[1] for item in paired]


def draw_loss_curve(
    timesteps: List[int],
    values: List[float],
    size: Tuple[int, int],
    y_range: Tuple[float, float],
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)

    margin_l, margin_r, margin_t, margin_b = 8, 5, 6, 11
    x0, y0 = margin_l, margin_t
    x1, y1 = width - margin_r - 1, height - margin_b - 1
    draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210))

    for frac in (0.25, 0.5, 0.75):
        y = y0 + int((y1 - y0) * frac)
        draw.line((x0, y, x1, y), fill=(232, 232, 232), width=1)
    for frac in (0.25, 0.5, 0.75):
        x = x0 + int((x1 - x0) * frac)
        draw.line((x, y0, x, y1), fill=(238, 238, 238), width=1)

    if not timesteps or not values:
        return image

    min_t, max_t = 1.0, 981.0
    min_y, max_y = y_range
    if max_y <= min_y:
        max_y = min_y + 1.0

    points = []
    for timestep, value in zip(timesteps, values):
        x = x0 + (float(timestep) - min_t) / (max_t - min_t) * (x1 - x0)
        y = y1 - (float(value) - min_y) / (max_y - min_y) * (y1 - y0)
        points.append((int(round(x)), int(round(y))))

    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(35, 97, 146))
    else:
        draw.line(points, fill=(35, 97, 146), width=2)
    return image


def compute_clip_image_score(
    gen_path: str,
    rec_path: str,
    clip_model,
    clip_preprocess,
    device: torch.device,
) -> float:
    with torch.no_grad():
        gen_image = clip_preprocess(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(device)
        rec_image = clip_preprocess(Image.open(rec_path).convert("RGB")).unsqueeze(0).to(device)
        gen_features = clip_model.encode_image(gen_image)
        rec_features = clip_model.encode_image(rec_image)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        rec_features = rec_features / rec_features.norm(dim=-1, keepdim=True)
        return float((gen_features @ rec_features.T).item())


def infer_uncond_root(rows: List[Dict[str, str]], args: argparse.Namespace) -> Path:
    if args.uncond_root:
        return Path(args.uncond_root)
    sample_id = int(rows[0]["sample_id"])
    return Path(f"outputs/uncond_slerp_initial_latents_sample{sample_id:04d}")


def load_uncond_image_map(uncond_root: Path) -> Dict[Tuple[str, int], str]:
    csv_path = uncond_root / "per_alpha_uncond_slerp_generation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing unconditional generation CSV: {csv_path}")
    image_map = {}
    for row in read_csv(csv_path):
        key = (row["pair_label"], int(row["alpha_index"]))
        image_map[key] = row["image_path"]
    return image_map


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    lines = text.split("\n")
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + max(0, len(lines) - 1) * 3
    x0, y0, x1, y1 = box
    y = y0 + max(0, (y1 - y0 - total_h) // 2)
    for line, width, height in zip(lines, line_widths, line_heights):
        x = x0 + max(0, (x1 - x0 - width) // 2)
        draw.text((x, y), line, font=font, fill=fill)
        y += height + 3


def make_group_montage(
    csv_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    clip_model,
    clip_preprocess,
    clip_device: torch.device,
) -> Path:
    rows = sorted(read_csv(csv_path), key=lambda row: float(row["alpha"]))
    rows = rows[args.start_index : args.start_index + args.num_columns]
    if len(rows) != args.num_columns:
        raise ValueError(
            f"{csv_path} has {len(rows)} selected rows, expected {args.num_columns}. "
            "Adjust --num_columns or --start_index."
        )

    group = csv_path.parent.name
    seed_a = rows[0]["seed_a"]
    seed_b = rows[0]["seed_b"]
    pair_label = rows[0]["pair_label"].replace("_", "-")
    uncond_root = infer_uncond_root(rows, args)
    uncond_image_map = load_uncond_image_map(uncond_root)

    cell = args.cell_size
    top_label_h = args.top_label_height
    metric_label_h = args.metric_label_height
    curve_h = args.curve_height
    left_w = args.left_label_width
    title_h = args.title_height
    row_gap = args.row_gap
    col_gap = args.col_gap
    pad = args.padding

    width = left_w + pad * 2 + args.num_columns * cell + (args.num_columns - 1) * col_gap
    height = (
        title_h
        + pad * 2
        + top_label_h
        + 4 * cell
        + curve_h
        + metric_label_h
        + 4 * row_gap
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(args.title_font_size)
    label_font = load_font(args.label_font_size)
    alpha_font = load_font(args.alpha_font_size)
    metric_font = load_font(args.metric_font_size)

    title = f"{group.replace('_', '-')} | {pair_label} | seed {seed_a} to seed {seed_b}"
    draw_centered_text(draw, (0, 0, width, title_h), title, title_font)

    x0 = left_w + pad
    y_alpha = title_h + pad
    y_gen = y_alpha + top_label_h
    y_rec = y_gen + cell + row_gap
    y_err = y_rec + cell + row_gap
    y_uncond = y_err + cell + row_gap
    y_curve = y_uncond + cell + row_gap
    y_metric = y_curve + curve_h

    row_labels = [
        ("alpha", y_alpha, top_label_h),
        ("Generated", y_gen, cell),
        ("Reconstructed", y_rec, cell),
        ("Error", y_err, cell),
        ("Uncond", y_uncond, cell),
        ("Inv. loss", y_curve, curve_h),
    ]
    for text, y, row_h in row_labels:
        draw_centered_text(draw, (pad, y, left_w, y + row_h), text, label_font)

    curves = [load_loss_curve(row) for row in rows]
    all_values = [value for _, values in curves for value in values]
    if all_values:
        y_range = (min(all_values), max(all_values))
    else:
        y_range = (0.0, 1.0)

    score_rows = []

    for col, row in enumerate(rows):
        x = x0 + col * (cell + col_gap)
        gen_image = resize_image(row["generated_image_path"], cell)
        rec_image = resize_image(row["reconstructed_image_path"], cell)
        err_image = make_error_image(gen_image, rec_image)
        uncond_key = (group, int(row["alpha_index"]))
        if uncond_key not in uncond_image_map:
            raise KeyError(f"Missing unconditional image for {uncond_key} under {uncond_root}")
        uncond_image = resize_image(uncond_image_map[uncond_key], cell)
        timesteps, neg_log_losses = curves[col]
        loss_curve = draw_loss_curve(timesteps, neg_log_losses, (cell, curve_h), y_range)
        clip_score = compute_clip_image_score(
            row["generated_image_path"],
            row["reconstructed_image_path"],
            clip_model,
            clip_preprocess,
            clip_device,
        )

        alpha = float(row["alpha"])
        draw_centered_text(draw, (x, y_alpha, x + cell, y_alpha + top_label_h), f"a={alpha:.1f}", alpha_font)
        canvas.paste(gen_image, (x, y_gen))
        canvas.paste(rec_image, (x, y_rec))
        canvas.paste(err_image, (x, y_err))
        canvas.paste(uncond_image, (x, y_uncond))
        canvas.paste(loss_curve, (x, y_curve))

        psnr = float(row["gen_rec_image_psnr"])
        text = f"PSNR={psnr:.2f}\nCLIP={clip_score:.4f}"
        draw_centered_text(
            draw,
            (x, y_metric, x + cell, y_metric + metric_label_h),
            text,
            metric_font,
        )
        score_rows.append(
            {
                "group": group,
                "pair_label": row["pair_label"],
                "alpha": f"{alpha:.4f}",
                "seed_a": row["seed_a"],
                "seed_b": row["seed_b"],
                "gen_rec_image_psnr": f"{psnr:.6f}",
                "gen_rec_clip_image_score": f"{clip_score:.8f}",
                "generated_image_path": row["generated_image_path"],
                "reconstructed_image_path": row["reconstructed_image_path"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / f"{group}_clip_scores_by_alpha.csv"
    with score_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    out_path = output_dir / f"{group}_gen_rec_error_uncond_loss_clip_5x{args.num_columns}.png"
    canvas.save(out_path)
    return out_path


def main(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    csv_paths = sorted(input_root.glob("*/per_alpha_slerp_fpi_metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per_alpha_slerp_fpi_metrics.csv found under {input_root}")

    clip_device = torch.device(args.device)
    clip_model, clip_preprocess = clip.load(args.clip_model, device=clip_device)
    clip_model.eval()

    for csv_path in csv_paths:
        out_path = make_group_montage(
            csv_path,
            output_dir,
            args,
            clip_model,
            clip_preprocess,
            clip_device,
        )
        print(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create generated/reconstructed/error/loss montages for SLERP FPI groups."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/slerp_fpi_gs7_seed_sensitive_sample0443",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/slerp_fpi_gs7_seed_sensitive_sample0443_metric_tables/montages_uncond_loss_clip_5x11",
    )
    parser.add_argument(
        "--uncond_root",
        type=str,
        default=None,
        help="Root of outputs from generate_uncond_slerp_initial_latents.py. "
        "Defaults to outputs/uncond_slerp_initial_latents_sampleXXXX.",
    )
    parser.add_argument("--num_columns", type=int, default=11)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--cell_size", type=int, default=150)
    parser.add_argument("--top_label_height", type=int, default=30)
    parser.add_argument("--metric_label_height", type=int, default=58)
    parser.add_argument("--curve_height", type=int, default=88)
    parser.add_argument("--left_label_width", type=int, default=118)
    parser.add_argument("--title_height", type=int, default=42)
    parser.add_argument("--padding", type=int, default=14)
    parser.add_argument("--row_gap", type=int, default=8)
    parser.add_argument("--col_gap", type=int, default=6)
    parser.add_argument("--title_font_size", type=int, default=17)
    parser.add_argument("--label_font_size", type=int, default=14)
    parser.add_argument("--alpha_font_size", type=int, default=16)
    parser.add_argument("--metric_font_size", type=int, default=15)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
