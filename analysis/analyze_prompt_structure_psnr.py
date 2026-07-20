#!/usr/bin/env python3
"""Heuristic prompt-structure analysis for FPI prompt PSNR tables."""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTICLES = {"a", "an", "the"}
LEADING_FILLERS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "several",
    "many",
    "some",
    "white",
    "black",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "brown",
    "gray",
    "grey",
    "small",
    "large",
    "little",
    "big",
    "round",
    "colorful",
}
STOP_HEADS = ARTICLES | LEADING_FILLERS | {"of", "and", "with", "to", "in", "on", "by", "near"}
NOUN_CUT_PREPS = {
    "of",
    "off",
    "with",
    "on",
    "in",
    "at",
    "near",
    "by",
    "under",
    "over",
    "behind",
    "beside",
    "against",
    "from",
    "into",
}
ANIMAL_WORDS = {
    "animal",
    "bear",
    "bird",
    "butterfly",
    "cat",
    "chipmunk",
    "cow",
    "dog",
    "duck",
    "elephant",
    "fish",
    "fox",
    "goat",
    "horse",
    "kitten",
    "koala",
    "lion",
    "meerkat",
    "moose",
    "mouse",
    "puppy",
    "rabbit",
    "rat",
    "retriever",
    "sheep",
    "squirrel",
    "swan",
    "wolf",
    "zebra",
}
PERSON_WORDS = {"boy", "child", "children", "girl", "man", "people", "person", "woman"}
VEHICLE_WORDS = {"airplane", "bicycle", "bike", "bus", "car", "motorcycle", "train", "truck"}
FOOD_WORDS = {
    "apple",
    "banana",
    "bowl",
    "cake",
    "coffee",
    "dumplings",
    "food",
    "fruit",
    "meat",
    "pizza",
    "plate",
    "sandwich",
    "steak",
}
ART_WORDS = {"art", "cartoon", "drawing", "painting", "photo", "picture", "statue"}
OUTDOOR_WORDS = {"beach", "field", "forest", "grass", "lake", "mountain", "ocean", "road", "sky", "street", "tree"}
INDOOR_WORDS = {"bed", "chair", "desk", "kitchen", "room", "sofa", "table"}

FRAME_PREFIXES = [
    ("a digital art of", "digital_art"),
    ("digital art of", "digital_art"),
    ("an anime painting of", "anime_painting"),
    ("anime painting of", "anime_painting"),
    ("a painting of", "painting"),
    ("painting of", "painting"),
    ("a photo of", "photo"),
    ("photo of", "photo"),
    ("a picture of", "picture"),
    ("picture of", "picture"),
    ("a drawing of", "drawing"),
    ("drawing of", "drawing"),
    ("a cartoon of", "cartoon"),
    ("cartoon of", "cartoon"),
    ("a portrait of", "portrait"),
    ("portrait of", "portrait"),
]

BE_ING_RE = re.compile(r"\b(is|are|was|were)\s+([a-z]+ing)\b")
BE_PART_RE = re.compile(r"\b(is|are|was|were)\s+([a-z]+ed|seen|made|covered|wrapped|dressed)\b")
GERUND_RE = re.compile(
    r"\b(sitting|standing|holding|laying|lying|walking|running|riding|flying|playing|"
    r"swimming|wearing|looking|leaning|jumping|sleeping|eating|drinking|driving|"
    r"floating|hanging|resting|reading|rising|crashing|peeking|smiling)\b"
)
FINITE_VERB_RE = re.compile(
    r"\b(sits|stands|holds|lays|lies|walks|runs|rides|flies|plays|swims|wears|"
    r"looks|leans|jumps|sleeps|hangs|crashes|floats)\b"
)
PARTICIPLE_RE = re.compile(r"\b(wrapped|covered|dressed|painted|made|seen|putted|placed)\b")
RELATION_PATTERNS = [
    ("on top of", re.compile(r"\bon top of\b")),
    ("in front of", re.compile(r"\bin front of\b")),
    ("next to", re.compile(r"\bnext to\b")),
    ("in the middle of", re.compile(r"\bin the middle of\b")),
    ("on", re.compile(r"\bon\b")),
    ("in", re.compile(r"\bin\b")),
    ("with", re.compile(r"\bwith\b")),
    ("near", re.compile(r"\bnear\b")),
    ("by", re.compile(r"\bby\b")),
    ("over", re.compile(r"\bover\b")),
    ("under", re.compile(r"\bunder\b")),
    ("behind", re.compile(r"\bbehind\b")),
    ("against", re.compile(r"\bagainst\b")),
    ("from", re.compile(r"\bfrom\b")),
    ("at", re.compile(r"\bat\b")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prompt structures against PSNR.")
    parser.add_argument(
        "--input_csv",
        default=str(PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "fpi_gs7_seed_psnr_prompt_mean_std.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "results" / "fpi_gs7_seed_psnr" / "prompt_structure_analysis"),
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def strip_frame_prefix(prompt: str) -> Tuple[str, str]:
    text = normalize(prompt)
    for prefix, frame_type in FRAME_PREFIXES:
        if text.startswith(prefix + " "):
            return text[len(prefix) :].strip(), frame_type
    return text, "none"


def choose_clause(prompt: str) -> str:
    parts = [part.strip() for part in prompt.split(",") if part.strip()]
    if len(parts) <= 1:
        return prompt
    scored = []
    for part in parts:
        score = 0
        if BE_ING_RE.search(part) or BE_PART_RE.search(part):
            score += 3
        if GERUND_RE.search(part) or FINITE_VERB_RE.search(part):
            score += 2
        score += len(tokens(part)) / 100.0
        scored.append((score, part))
    return max(scored)[1]


def find_predicate(text: str) -> Tuple[str, str, int, int]:
    candidates = []
    for pred_type, regex in [
        ("be_gerund", BE_ING_RE),
        ("be_participle", BE_PART_RE),
        ("gerund_modifier", GERUND_RE),
        ("finite_verb", FINITE_VERB_RE),
        ("participle_modifier", PARTICIPLE_RE),
    ]:
        match = regex.search(text)
        if match:
            verb = match.group(2) if pred_type.startswith("be_") else match.group(1)
            candidates.append((match.start(), match.end(), pred_type, verb))
    if candidates:
        start, end, pred_type, verb = sorted(candidates, key=lambda item: item[0])[0]
        return pred_type, verb, start, end

    relation = find_relation(text)
    if relation:
        rel, start, end, _ = relation
        return "prepositional_np", rel, start, end
    return "noun_phrase", "", len(text), len(text)


def find_relation(text: str) -> Optional[Tuple[str, int, int, str]]:
    best = None
    for relation, regex in RELATION_PATTERNS:
        match = regex.search(text)
        if not match:
            continue
        if best is None or match.start() < best[1]:
            best = (relation, match.start(), match.end(), text[match.end() :].strip())
    return best


def clean_np(text: str) -> str:
    text = normalize(text)
    words = tokens(text)
    while words and words[0] in ARTICLES:
        words.pop(0)
    return " ".join(words)


def head_of_phrase(phrase: str) -> str:
    words = tokens(phrase)
    if not words:
        return ""
    cut_words = []
    for word in words:
        if word in NOUN_CUT_PREPS and cut_words:
            break
        cut_words.append(word)
    for word in reversed(cut_words):
        if word not in STOP_HEADS:
            return word
    return cut_words[-1] if cut_words else words[-1]


def categorize(head: str, phrase: str, prompt: str) -> str:
    phrase_words = set(tokens(" ".join([head, phrase])))
    prompt_words = set(tokens(prompt))
    if head in ANIMAL_WORDS or phrase_words & ANIMAL_WORDS:
        return "animal"
    if head in PERSON_WORDS or phrase_words & PERSON_WORDS:
        return "person"
    if head in VEHICLE_WORDS or phrase_words & VEHICLE_WORDS:
        return "vehicle"
    if head in FOOD_WORDS or phrase_words & FOOD_WORDS:
        return "food"
    if head in ART_WORDS:
        return "art_object"
    if head in INDOOR_WORDS:
        return "indoor_object"
    if head in OUTDOOR_WORDS:
        return "outdoor_scene"
    if prompt_words & OUTDOOR_WORDS:
        return "outdoor_scene"
    if prompt_words & INDOOR_WORDS:
        return "indoor_object"
    if prompt_words & FOOD_WORDS:
        return "food"
    return "other"


def object_from_relation(predicate_tail: str) -> Tuple[str, str, str]:
    relation = find_relation(predicate_tail)
    if not relation:
        return "", "", ""
    rel, _, _, object_phrase = relation
    return rel, clean_np(object_phrase), head_of_phrase(object_phrase)


def syntax_pattern(predicate_type: str, relation: str, frame_type: str) -> str:
    relation_part = relation.replace(" ", "_") if relation else "no_relation"
    if frame_type != "none":
        return f"{frame_type}+{predicate_type}+{relation_part}"
    return f"{predicate_type}+{relation_part}"


def analyze_prompt(prompt: str) -> Dict[str, str]:
    normalized = normalize(prompt)
    clause = choose_clause(normalized)
    semantic_text, frame_type = strip_frame_prefix(clause)
    pred_type, verb, start, end = find_predicate(semantic_text)

    subject_phrase = clean_np(semantic_text[:start])
    if not subject_phrase and pred_type == "noun_phrase":
        subject_phrase = clean_np(semantic_text)
    subject_head = head_of_phrase(subject_phrase)

    predicate_tail = semantic_text[start:]
    relation, object_phrase, object_head = object_from_relation(predicate_tail)
    subject_category = categorize(subject_head, subject_phrase, normalized)

    return {
        "normalized_prompt": normalized,
        "semantic_clause": semantic_text,
        "frame_type": frame_type,
        "subject_phrase": subject_phrase,
        "subject_head": subject_head,
        "subject_category": subject_category,
        "predicate_type": pred_type,
        "predicate_verb": verb,
        "relation": relation or "",
        "object_phrase": object_phrase,
        "object_head": object_head,
        "syntax_pattern": syntax_pattern(pred_type, relation, frame_type),
        "word_count": len(tokens(normalized)),
        "comma_count": normalized.count(","),
    }


def read_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            row["prompt_id"] = idx
            row["psnr_mean"] = float(row["psnr_mean"])
            row["psnr_std"] = float(row["psnr_std"])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[Dict], key: str) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key] or "none"].append(row)

    out = []
    for value, group_rows in grouped.items():
        psnr = [float(row["psnr_mean"]) for row in group_rows]
        seed_std = [float(row["psnr_std"]) for row in group_rows]
        out.append(
            {
                key: value,
                "n": len(group_rows),
                "psnr_mean_avg": mean(psnr),
                "psnr_mean_std": stdev(psnr) if len(psnr) > 1 else 0.0,
                "psnr_mean_min": min(psnr),
                "psnr_mean_max": max(psnr),
                "seed_psnr_std_avg": mean(seed_std),
            }
        )
    return sorted(out, key=lambda row: (-int(row["n"]), float(row["psnr_mean_avg"])))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def md_table(rows: List[Dict], columns: List[Tuple[str, str]], limit: Optional[int] = None) -> str:
    rows = rows[:limit] if limit else rows
    header = "| " + " | ".join(label for _, label in columns) + " |"
    align = "| " + " | ".join("---:" if key in {"n"} or "psnr" in key else "---" for key, _ in columns) + " |"
    lines = [header, align]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, feature_rows: List[Dict], summaries: Dict[str, List[Dict]]) -> None:
    sorted_by_psnr = sorted(feature_rows, key=lambda row: float(row["psnr_mean"]))
    squirrel_rows = [row for row in feature_rows if "squirrel" in row["normalized_prompt"]]
    animal_rows = [row for row in feature_rows if row["subject_category"] == "animal"]

    lines = [
        "# Prompt Structure PSNR Analysis",
        "",
        "This report uses lightweight heuristic parsing rather than a full dependency parser.",
        f"Total prompts: {len(feature_rows)}",
        "",
        "## Lowest PSNR Prompts",
        "",
        md_table(
            sorted_by_psnr[:15],
            [
                ("prompt_id", "id"),
                ("original_prompt", "prompt"),
                ("subject_head", "subject"),
                ("predicate_type", "predicate"),
                ("relation", "relation"),
                ("object_head", "object"),
                ("psnr_mean", "PSNR mean"),
                ("psnr_std", "PSNR std"),
            ],
        ),
        "",
        "## Squirrel Prompts",
        "",
        md_table(
            squirrel_rows,
            [
                ("prompt_id", "id"),
                ("original_prompt", "prompt"),
                ("subject_head", "subject"),
                ("predicate_type", "predicate"),
                ("relation", "relation"),
                ("object_head", "object"),
                ("psnr_mean", "PSNR mean"),
                ("psnr_std", "PSNR std"),
            ],
        ),
        "",
        "## Summary By Subject Category",
        "",
        md_table(
            summaries["subject_category"],
            [
                ("subject_category", "category"),
                ("n", "n"),
                ("psnr_mean_avg", "avg mean PSNR"),
                ("psnr_mean_min", "min mean PSNR"),
                ("psnr_mean_max", "max mean PSNR"),
                ("seed_psnr_std_avg", "avg seed std"),
            ],
        ),
        "",
        "## Summary By Predicate Type",
        "",
        md_table(
            summaries["predicate_type"],
            [
                ("predicate_type", "predicate"),
                ("n", "n"),
                ("psnr_mean_avg", "avg mean PSNR"),
                ("psnr_mean_min", "min mean PSNR"),
                ("psnr_mean_max", "max mean PSNR"),
                ("seed_psnr_std_avg", "avg seed std"),
            ],
        ),
        "",
        "## Summary By Relation",
        "",
        md_table(
            summaries["relation"],
            [
                ("relation", "relation"),
                ("n", "n"),
                ("psnr_mean_avg", "avg mean PSNR"),
                ("psnr_mean_min", "min mean PSNR"),
                ("psnr_mean_max", "max mean PSNR"),
                ("seed_psnr_std_avg", "avg seed std"),
            ],
        ),
        "",
        "## Frequent Subject Heads",
        "",
        md_table(
            [row for row in summaries["subject_head"] if int(row["n"]) >= 3][:30],
            [
                ("subject_head", "subject"),
                ("n", "n"),
                ("psnr_mean_avg", "avg mean PSNR"),
                ("psnr_mean_min", "min mean PSNR"),
                ("psnr_mean_max", "max mean PSNR"),
                ("seed_psnr_std_avg", "avg seed std"),
            ],
        ),
        "",
        "## Animal Prompts: Lowest PSNR",
        "",
        md_table(
            sorted(animal_rows, key=lambda row: float(row["psnr_mean"]))[:20],
            [
                ("prompt_id", "id"),
                ("original_prompt", "prompt"),
                ("subject_head", "subject"),
                ("predicate_type", "predicate"),
                ("relation", "relation"),
                ("object_head", "object"),
                ("psnr_mean", "PSNR mean"),
                ("psnr_std", "PSNR std"),
            ],
        ),
        "",
    ]
    (output_dir / "prompt_structure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    feature_rows = []
    for row in rows:
        features = analyze_prompt(row["original_prompt"])
        feature_rows.append({**row, **features})

    feature_fields = [
        "prompt_id",
        "original_prompt",
        "psnr_mean",
        "psnr_std",
        "normalized_prompt",
        "semantic_clause",
        "frame_type",
        "subject_phrase",
        "subject_head",
        "subject_category",
        "predicate_type",
        "predicate_verb",
        "relation",
        "object_phrase",
        "object_head",
        "syntax_pattern",
        "word_count",
        "comma_count",
    ]
    write_csv(output_dir / "prompt_structure_features.csv", feature_rows, feature_fields)

    summary_keys = [
        "subject_category",
        "subject_head",
        "predicate_type",
        "predicate_verb",
        "relation",
        "object_head",
        "syntax_pattern",
        "frame_type",
        "word_count",
    ]
    summaries = {}
    for key in summary_keys:
        summary = summarize(feature_rows, key)
        summaries[key] = summary
        write_csv(output_dir / f"summary_by_{key}.csv", summary)

    write_report(output_dir, feature_rows, summaries)
    print(f"Wrote prompt-structure analysis for {len(feature_rows)} prompts to: {output_dir}")


if __name__ == "__main__":
    main()
