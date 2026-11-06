#!/usr/bin/env python3
"""
Export 5,000 MIMIC-CXR mortality cohort images pre-resized to 224x224 letterbox.
Mathematically identical to runtime LetterboxResize, but compresses 8.2 GB -> ~60 MB.
"""

import csv
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/mimic_cxr"
MANIFEST_PATH = DATA_ROOT / "manifest_mortality.csv"
OUT_ROOT = ROOT / "data/mimic_cxr_224"
OUT_ZIP = ROOT / "data/mimic_cxr_224.zip"


def process_image(rel_path: str):
    src_file = DATA_ROOT / rel_path
    dst_file = OUT_ROOT / rel_path

    if not src_file.exists():
        return False, f"Missing: {src_file}"

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_file) as img:
        img_rgb = img.convert("RGB")
        # Exact letterbox resize: pad with black to 224x224 maintaining aspect ratio
        resized = ImageOps.pad(img_rgb, (224, 224), method=Image.Resampling.BILINEAR, color=0)
        resized.save(dst_file, format="JPEG", quality=92, optimize=True)

    return True, None


def main():
    start = time.time()
    print(f"Reading manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rel_paths = [r["relative_path"] for r in rows]
    total = len(rel_paths)
    print(f"Total images to process: {total}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy manifest
    shutil.copy2(MANIFEST_PATH, OUT_ROOT / "manifest_mortality.csv")
    print(f"Copied manifest to {OUT_ROOT / 'manifest_mortality.csv'}")

    workers = min(8, os.cpu_count() or 4)
    print(f"Processing with {workers} workers...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (ok, err) in enumerate(executor.map(process_image, rel_paths), 1):
            if ok:
                success_count += 1
            else:
                print(f"Error: {err}")
            if i % 500 == 0 or i == total:
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")

    elapsed = time.time() - start
    print(f"Resized {success_count}/{total} images in {elapsed:.2f}s")

    # Zip the exported directory
    print(f"Zipping {OUT_ROOT} to {OUT_ZIP}...")
    zip_start = time.time()
    shutil.make_archive(str(OUT_ROOT), "zip", root_dir=str(OUT_ROOT.parent), base_dir=OUT_ROOT.name)
    zip_size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Created {OUT_ZIP} ({zip_size_mb:.2f} MB) in {time.time() - zip_start:.2f}s")


if __name__ == "__main__":
    main()
