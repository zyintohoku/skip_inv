#!/usr/bin/env python3
"""
Generate merge_with_images outputs for all AIDI-GS7 samples.
Output directory: results/aidi_gs7_merge_with_images_all
"""

import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = "aidi_gs7"
OUTPUT_BASE = "outputs/reconstruction"
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "aidi_gs7_merge_with_images_all")
CONV_PATH = os.path.join(PROJECT_ROOT, OUTPUT_BASE, METHOD, "convergence_losses.json")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CONV_PATH, "r", encoding="utf-8") as f:
        conv = json.load(f)
    sample_ids = [int(item["sample_id"]) for item in conv]

    print(f"Total samples: {len(sample_ids)}")
    failed = []
    for i, sid in enumerate(sample_ids, 1):
        save_path = os.path.join(OUT_DIR, f"convergence_plot_{METHOD}_s{sid}.png")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "analysis", "plot_convergence.py"),
            "--sample_ids",
            str(sid),
            "--method",
            METHOD,
            "--merge_with_images",
            "--output_base",
            OUTPUT_BASE,
            "--save",
            save_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            failed.append((sid, proc.stdout[-300:]))
        if i % 50 == 0 or i == len(sample_ids):
            print(f"Processed {i}/{len(sample_ids)}")

    print(f"Output dir: {OUT_DIR}")
    print(f"Generated: {len(sample_ids) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed sample IDs:", [x[0] for x in failed])


if __name__ == "__main__":
    main()
