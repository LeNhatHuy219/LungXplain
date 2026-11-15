"""Read only the validated exact-5000 cohort and its declared predictors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/mimic_cxr/multimodal_v3"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class Cohort:
    frame: pd.DataFrame
    schema: dict
    manifest_hash: str
    dataset_dir: Path

    @property
    def concept_names(self):
        return list(self.schema["concept_label_columns"])

    def clinical(self):
        columns = self.schema["model_clinical_columns"]
        result = self.frame[columns].copy()
        categorical = set(self.schema["categorical_value_columns"])
        for name in columns:
            if name in categorical:
                result[name] = result[name].replace("", np.nan)
            else:
                result[name] = pd.to_numeric(result[name].replace("", np.nan), errors="raise").astype(float)
        return result

    def concepts(self):
        return self.frame[self.concept_names].to_numpy(dtype=np.float32)

    def known(self):
        return self.frame[["known_" + name for name in self.concept_names]].to_numpy(dtype=np.int8).astype(bool)

    def outcome(self):
        return self.frame[self.schema["outcome_column"]].to_numpy(dtype=np.int64)

    def indices(self, split):
        if split not in {"train", "valid", "test"}:
            raise ValueError("Unknown dataset split")
        return np.flatnonzero(self.frame["split"].to_numpy() == split)


def load_cohort(dataset_dir=DEFAULT_DATASET, *, sensitivity=False):
    directory = Path(dataset_dir).resolve()
    validation_path = directory / "validation_report.json"
    if not validation_path.is_file():
        raise ValueError("Dataset has no final validation report; complete the exact-5000 dataset first")
    validation = json.loads(validation_path.read_text())
    if not (validation.get("passed") is True and validation.get("active_rows") == 5000
            and validation.get("exact_5000_validation", {}).get("valid_images") == 5000):
        raise ValueError("ACC-ECBM requires a validated cohort with exactly 5000 valid images")
    if sha256(directory / "report.json") != validation["report_sha256"]:
        raise ValueError("Dataset report changed after validation")
    report = json.loads((directory / "report.json").read_text())
    for name in ("feature_schema.json", "manifest.csv", "manifest_temporal_sensitivity.csv"):
        if sha256(directory / name) != report["output_sha256"][name]:
            raise ValueError("Dataset artifact changed after validation: " + name)
    name = "manifest_temporal_sensitivity.csv" if sensitivity else "manifest.csv"
    expected = validation["sensitivity_manifest_sha256" if sensitivity else "primary_manifest_sha256"]
    actual = sha256(directory / name)
    if actual != expected:
        raise ValueError("Manifest checksum does not match final validation")
    frame = pd.read_csv(directory / name, dtype=str, keep_default_na=False)
    if len(frame) != 5000 or any(frame[key].nunique() != 5000 for key in ("subject_id", "hadm_id", "dicom_id")):
        raise ValueError("Cohort does not contain exactly 5000 distinct patients/admissions/images")
    if frame["split"].value_counts().to_dict() != {"train": 4000, "valid": 500, "test": 500}:
        raise ValueError("Unexpected patient split")
    if not all((ROOT / name).is_file() for name in frame["image_path"]):
        raise ValueError("A source image is missing; no placeholder images are permitted")
    schema = json.loads((directory / "feature_schema.json").read_text())
    if len(schema["concept_label_columns"]) != 14:
        raise ValueError("Expected the 14 CheXpert concepts")
    if set(schema["model_clinical_columns"]) & {"diagnosis_binary", "subject_id", "hadm_id", "split", "prediction_time", "deathtime", "dischtime"}:
        raise ValueError("Clinical allowlist contains a forbidden predictor")
    return Cohort(frame, schema, actual, directory)
