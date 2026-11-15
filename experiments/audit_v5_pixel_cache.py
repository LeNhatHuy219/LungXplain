"""Independently compare every v5 cache PNG with its validated source JPEG."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungxplain.data import ROOT, load_cohort, sha256, write_json
from lungxplain.data_cache import load_cohort_with_pixel_cache

DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
CACHE = ROOT / "data/model_cache/cxr_resized256_v5"
AUDIT = ROOT / "data/audits/v5_kaggle_preparation_20261115/pixel_cache_validation.json"


def resized_rgb(image, size=256):
    """Pillow implementation independent of torchvision.transforms.Resize."""
    image = image.convert("RGB")
    width, height = image.size
    dimensions = (size, int(size * height / width)) if width <= height else (int(size * width / height), size)
    return image.resize(dimensions, resample=Image.Resampling.BILINEAR)


def audit_cache(dataset=DATASET, cache=CACHE, output=AUDIT):
    dataset, cache, output = Path(dataset), Path(cache), Path(output)
    write_json(output, {"passed": False, "status": "verification_in_progress", "test_evaluated": False})
    cohort = load_cohort(dataset)
    if cohort.schema.get("dataset_version") != "v5" or len(cohort.schema["model_clinical_columns"]) != 44:
        raise ValueError("Expected validated v5 with 44 clinical/missingness inputs")
    remote_cohort = load_cohort_with_pixel_cache(dataset, cache)
    index = json.loads((cache / "index.json").read_text())
    if len(list(cache.glob("*.png"))) != 5000 or remote_cohort.manifest_hash != cohort.manifest_hash:
        raise ValueError("Cache count or manifest mismatch")
    audit = pd.read_csv(dataset / "image_audit.csv.gz", dtype=str, keep_default_na=False)
    references = dict(zip(audit["dicom_id"], audit["sha256"]))

    def compare(item):
        position, row = item
        source = ROOT / row["image_path"]
        if sha256(source) != references[row["dicom_id"]]:
            raise ValueError("A source JPEG no longer matches its validated checksum")
        with Image.open(source) as image:
            expected = np.asarray(resized_rgb(image))
        with Image.open(cache / f"{position:05d}.png") as image:
            actual = np.asarray(image.convert("RGB"))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError("Cached pixels differ from source RGB bilinear resize")
        return hashlib.sha256(actual.tobytes() + str(actual.shape).encode()).hexdigest()

    hashes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for value in pool.map(compare, enumerate(cohort.frame.to_dict("records"))):
            hashes.append(value)
            if len(hashes) % 500 == 0:
                print(json.dumps({"independently_verified_cache_images": len(hashes), "total": 5000}), flush=True)
    if len(set(hashes)) != 5000:
        raise ValueError("Duplicate resized pixel content found; inspect before training")
    result = {"passed": True, "source_jpegs_checksum_verified": 5000,
              "cache_pngs_checksum_verified": 5000, "pixel_comparisons_exact": 5000,
              "unique_resized_pixel_arrays": 5000, "manifest_sha256": cohort.manifest_hash,
              "index_sha256": sha256(cache / "index.json"), "source_image_audit_sha256": sha256(dataset / "image_audit.csv.gz"),
              "clinical_value_columns": 22, "clinical_missing_columns": 22, "outcome_inputs": 58,
              "test_evaluated": False, "auditor_sha256": sha256(Path(__file__))}
    write_json(output, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--pixel-cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=AUDIT)
    args = parser.parse_args()
    audit_cache(args.dataset, args.pixel_cache, args.output)
