#!/usr/bin/env python3
"""Build a gen/rec montage for selected modifier-context prompt-grid cells."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_DIR = (
    PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "modifier_prompt_grid_fpi" / "merged_contexts"
)
DEFAULT_NAME = "modifier_prompt_grid_merged_context_fpi"
CONTEXT_DISPLAY = {
    "bare": "bare",
    "wooden_fence": "wooden fence",
    "wooden_table": "wooden table",
    "tree_branch": "tree branch",
    "rock": "rock",
    "grass": "grass",
    "shallow_water": "shallow water",
    "snowy_field": "snowy field",
    "transparent_glass_jar": "glass jar",
    "streetlight_night": "streetlight night",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a subject x context montage where each cell stacks generated and reconstructed images."
    )
    parser.add_argument("--merged_dir", default=str(DEFAULT_MERGED_DIR))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--row_stride",
        type=int,
        default=2,
        help="Take every Nth heatmap subject row after --row_start.",
    )
    parser.add_argument(
        "--row_start",
        type=int,
        default=1,
        help="1-based heatmap row to start from. Default 1 selects odd rows with row_stride=2.",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--pair_gap", type=int, default=3)
    parser.add_argument("--left_label_width", type=int, default=128)
    parser.add_argument("--bottom_label_height", type=int, default=118)
    parser.add_argument("--subject_font_size", type=int, default=22)
    parser.add_argument("--context_font_size", type=int, default=22)
    parser.add_argument("--context_angle", type=float, default=35.0)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--background", default="white")
    parser.add_argument("--output", default="")
    parser.add_argument("--manifest", default="")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def heatmap_order(mean_matrix_path: Path, row_start: int, row_stride: int) -> tuple[list[str], list[str]]:
    rows = read_csv(mean_matrix_path)
    if not rows:
        raise ValueError(f"No rows found in {mean_matrix_path}")

    contexts = [
        key
        for key in rows[0].keys()
        if key not in {"label", "subject_psnr_mean", "subject_psnr_mean_std", "subject_n"}
    ]
    labels = [row["label"] for row in rows if row.get("label") and not row["label"].startswith("context_psnr")]
    start = max(row_start - 1, 0)
    return labels[start::row_stride], contexts


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


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
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=3, align="center")


def draw_rotated_centered(
    canvas: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    angle: float,
    fill: tuple[int, int, int] = (28, 28, 28),
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 10
    label = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(*fill, 255))
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    rotated = label.rotate(angle, expand=True, resample=resample)
    x = int(center[0] - rotated.width / 2)
    y = int(center[1] - rotated.height / 2)
    canvas.paste(rotated, (x, y), rotated)


def wrap_label(text: str, max_words_per_line: int = 2) -> str:
    words = text.split()
    if len(words) <= max_words_per_line:
        return text
    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i : i + max_words_per_line]))
    return "\n".join(lines)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "col",
        "label",
        "context",
        "prompt_id",
        "source_result_set",
        "source_prompt_id",
        "prompt",
        "gen_image_path",
        "rec_image_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_montage(args: argparse.Namespace) -> Path:
    merged_dir = Path(args.merged_dir)
    detail_path = merged_dir / f"{args.name}_psnr_detail.csv"
    mean_matrix_path = merged_dir / f"{args.name}_class_context_mean_matrix.csv"
    labels, contexts = heatmap_order(mean_matrix_path, args.row_start, args.row_stride)

    detail_rows = read_csv(detail_path)
    by_cell = {}
    for row in detail_rows:
        if row.get("seed") != str(args.seed):
            continue
        by_cell[(row.get("label", ""), row.get("context", ""))] = row

    missing = [(label, context) for label in labels for context in contexts if (label, context) not in by_cell]
    if missing:
        preview = ", ".join(f"{label}/{context}" for label, context in missing[:8])
        raise FileNotFoundError(f"Missing {len(missing)} seed-{args.seed} cells: {preview}")

    cell_w = args.image_size
    cell_h = args.image_size * 2 + args.pair_gap
    grid_w = len(contexts) * cell_w + (len(contexts) - 1) * args.gap
    grid_h = len(labels) * cell_h + (len(labels) - 1) * args.gap
    width = args.margin * 2 + args.left_label_width + args.gap + grid_w
    height = args.margin * 2 + grid_h + args.gap + args.bottom_label_height

    canvas = Image.new("RGB", (width, height), args.background)
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(args.subject_font_size, bold=True)
    context_font = load_font(args.context_font_size, bold=True)
    stroke = (218, 218, 218)

    grid_left = args.margin + args.left_label_width + args.gap
    grid_top = args.margin
    manifest_rows = []

    for row_idx, label in enumerate(labels):
        y = grid_top + row_idx * (cell_h + args.gap)
        draw_centered(
            draw,
            (args.margin, y, args.margin + args.left_label_width, y + cell_h),
            label,
            label_font,
        )
        for col_idx, context in enumerate(contexts):
            x = grid_left + col_idx * (cell_w + args.gap)
            row = by_cell[(label, context)]
            gen_path = resolve_project_path(row["gen_image_path"])
            rec_path = resolve_project_path(row["rec_image_path"])
            if not gen_path.exists() or not rec_path.exists():
                raise FileNotFoundError(f"Missing image pair for {label}/{context}: {gen_path}, {rec_path}")

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
            manifest_rows.append(
                {
                    "row": row_idx + 1,
                    "col": col_idx + 1,
                    "label": label,
                    "context": context,
                    "prompt_id": row.get("prompt_id", ""),
                    "source_result_set": row.get("source_result_set", ""),
                    "source_prompt_id": row.get("source_prompt_id", ""),
                    "prompt": row.get("prompt", ""),
                    "gen_image_path": row.get("gen_image_path", ""),
                    "rec_image_path": row.get("rec_image_path", ""),
                }
            )

    context_top = grid_top + grid_h + args.gap
    for col_idx, context in enumerate(contexts):
        x = grid_left + col_idx * (cell_w + args.gap)
        display = CONTEXT_DISPLAY.get(context, context.replace("_", " "))
        draw_rotated_centered(
            canvas,
            (x + cell_w // 2, context_top + args.bottom_label_height // 2),
            display,
            context_font,
            args.context_angle,
        )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = merged_dir / f"{args.name}_seed{args.seed}_odd_subject_gen_rec_montage.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".csv")
    write_manifest(manifest_path, manifest_rows)
    return output_path


if __name__ == "__main__":
    print(make_montage(parse_args()))
