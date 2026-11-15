"""Load the validated exact-5000 cohort where only the resized pixel cache is present.

Used for remote GPU runs (private Kaggle) so the 8.9 GB source JPEGs need not be
uploaded. Every dataset check of ``data.load_cohort`` is repeated; the source-JPEG
existence check is replaced by a stronger proof: the pixel cache index must be
bound to this manifest and to the validated ``image_audit.csv.gz`` (source JPEG
checksums), contain exactly 5000 images in manifest order, and every PNG must
match its recorded SHA-256. ``data.py`` is not edited (v1 hash binding).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import DEFAULT_DATASET, ROOT, Cohort, sha256

FORBIDDEN = {"diagnosis_binary", "subject_id", "hadm_id", "split", "prediction_time", "deathtime", "dischtime"}


def load_cohort_with_pixel_cache(dataset_dir=DEFAULT_DATASET, pixel_cache=ROOT / "data/model_cache/cxr_resized256_v3",
                                 *, resize_size=256):
    directory = Path(dataset_dir).resolve()
    validation = json.loads((directory / "validation_report.json").read_text())
    if not (validation.get("passed") is True and validation.get("active_rows") == 5000
            and validation.get("exact_5000_validation", {}).get("valid_images") == 5000):
        raise ValueError("ACC-ECBM requires a validated cohort with exactly 5000 valid images")
    if sha256(directory / "report.json") != validation["report_sha256"]:
        raise ValueError("Dataset report changed after validation")
    report = json.loads((directory / "report.json").read_text())
    for name in ("feature_schema.json", "manifest.csv", "manifest_temporal_sensitivity.csv"):
        if sha256(directory / name) != report["output_sha256"][name]:
            raise ValueError("Dataset artifact changed after validation: " + name)
    actual = sha256(directory / "manifest.csv")
    if actual != validation["primary_manifest_sha256"]:
        raise ValueError("Manifest checksum does not match final validation")
    frame = pd.read_csv(directory / "manifest.csv", dtype=str, keep_default_na=False)
    if len(frame) != 5000 or any(frame[key].nunique() != 5000 for key in ("subject_id", "hadm_id", "dicom_id")):
        raise ValueError("Cohort does not contain exactly 5000 distinct patients/admissions/images")
    if frame["split"].value_counts().to_dict() != {"train": 4000, "valid": 500, "test": 500}:
        raise ValueError("Unexpected patient split")
    schema = json.loads((directory / "feature_schema.json").read_text())
    if len(schema["concept_label_columns"]) != 14:
        raise ValueError("Expected the 14 CheXpert concepts")
    if set(schema["model_clinical_columns"]) & FORBIDDEN:
        raise ValueError("Clinical allowlist contains a forbidden predictor")

    cache = Path(pixel_cache).resolve()
    index = json.loads((cache / "index.json").read_text())
    expected = {"manifest_sha256": actual, "resize_size": resize_size, "interpolation": "PIL_RGB_BILINEAR",
                "learned_features": False, "source_image_audit_sha256": sha256(directory / "image_audit.csv.gz")}
    if index["configuration"] != expected or index["rows"] != 5000 or len(index["images"]) != 5000:
        raise ValueError("Pixel cache is not bound to this validated manifest and image audit")
    for position, record in enumerate(index["images"]):
        if record["file"] != f"{position:05d}.png" or sha256(cache / record["file"]) != record["sha256"]:
            raise ValueError("Resized pixel cache is incomplete or modified")
    return Cohort(frame, schema, actual, directory)
