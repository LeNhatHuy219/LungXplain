"""Load a classification cohort without imputing unmentioned/uncertain labels."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

SPLITS = ("train", "validate", "test")
TARGET = "Pneumonia"
LABEL_POLICY = "explicit_binary_only"


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path.name}: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty table: {path}")
    return rows


@dataclass
class ClassificationCohort:
    root: Path
    rows: list[dict]
    audit: list[dict]
    report: dict

    def split(self, name: str) -> list[dict]:
        if name not in SPLITS:
            raise ValueError(f"Unknown split: {name}")
        return [row for row in self.rows if row["split"] == name]


def load_classification_cohort(data_root: str | Path) -> ClassificationCohort:
    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "selection_manifest.csv"
    labels_path = root / "selected_chexpert_labels.csv"
    manifest = read_csv(manifest_path, {"dicom_id", "subject_id", "study_id", "split", "relative_path"})
    labels = read_csv(labels_path, {"subject_id", "study_id", TARGET})

    # Validate leakage on the entire manifest, including later-excluded samples.
    patient_split: dict[str, str] = {}
    seen_images, seen_studies, seen_paths = set(), set(), set()
    for row in manifest:
        if any(not row[key] for key in ("subject_id", "study_id", "dicom_id", "relative_path")):
            raise ValueError("Manifest contains an empty identifier or image path")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unexpected manifest split: {split}")
        patient = row["subject_id"]
        if patient in patient_split and patient_split[patient] != split:
            raise ValueError("Patient overlap between manifest splits")
        patient_split[patient] = split
        study = (patient, row["study_id"])
        if row["dicom_id"] in seen_images or study in seen_studies or row["relative_path"] in seen_paths:
            raise ValueError("Duplicate image, study, or path in selection manifest")
        seen_images.add(row["dicom_id"])
        seen_studies.add(study)
        seen_paths.add(row["relative_path"])
        image_path = root / row["relative_path"]
        if not image_path.resolve().is_relative_to(root):
            raise ValueError("Manifest image path escapes the dataset directory")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing manifest image: {image_path}")

    label_index = {}
    for row in labels:
        key = (row["subject_id"], row["study_id"])
        if key in label_index:
            raise ValueError("Duplicate subject/study in label table")
        label_index[key] = row

    included, audit = [], []
    counts = {split: Counter(total=0, included=0, positive=0, negative=0,
                             excluded_uncertain=0, excluded_not_mentioned=0) for split in SPLITS}
    for row in manifest:
        key = (row["subject_id"], row["study_id"])
        if key not in label_index:
            raise ValueError("Manifest study is missing from the label table")
        raw = label_index[key][TARGET].strip()
        counts[row["split"]]["total"] += 1
        record = dict(row, target_raw=raw, target="", status="")
        if raw in {"0", "0.0", "1", "1.0"}:
            target = int(float(raw))
            record.update(target=target, status="included")
            included.append(record)
            counts[row["split"]]["included"] += 1
            counts[row["split"]]["positive" if target else "negative"] += 1
        elif raw in {"-1", "-1.0"}:
            record["status"] = "excluded_uncertain"
            counts[row["split"]]["excluded_uncertain"] += 1
        elif raw == "":
            record["status"] = "excluded_not_mentioned"
            counts[row["split"]]["excluded_not_mentioned"] += 1
        else:
            raise ValueError(f"Unexpected raw {TARGET} value: {raw!r}")
        audit.append(record)

    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (manifest_path, labels_path)}
    fingerprint = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(source_hashes.items())).encode()
    ).hexdigest()
    report = {
        "target": TARGET,
        "target_classes": {"0": "Non-Pneumonia", "1": "Pneumonia"},
        "label_policy": LABEL_POLICY,
        "label_source": labels_path.name,
        "label_provenance": "Automatic CheXpert extraction from radiology reports; not image-level expert confirmation.",
        "blank_labels_imputed": False,
        "uncertain_labels_imputed": False,
        "concepts_required": False,
        "original_manifest_count": len(manifest),
        "included_count": len(included),
        "counts_by_split": {split: dict(value) for split, value in counts.items()},
        "patient_disjoint": True,
        "patients_by_split": {split: len({r["subject_id"] for r in included if r["split"] == split})
                              for split in SPLITS},
        "source_sha256": source_hashes,
        "data_fingerprint": fingerprint,
    }
    return ClassificationCohort(root, included, audit, report)


class LetterboxResize:
    """Preserve the complete image and aspect ratio using black padding."""

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("Image size must be positive")
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR, color=0)


def image_transforms(image_size: int = 224, training: bool = False):
    steps = [LetterboxResize(image_size)]
    if training:
        steps += [transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10, fill=0)]
    return transforms.Compose(steps + [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class MimicCxrClassificationDataset(Dataset):
    """Image -> diagnosis only. No concept labels are read or required."""

    def __init__(self, cohort: ClassificationCohort, split: str, image_size: int = 224):
        self.root = cohort.root
        self.rows = cohort.split(split)
        if not self.rows:
            raise ValueError(f"No included samples in {split}")
        self.transform = image_transforms(image_size, training=split == "train")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(row["target"], dtype=torch.long),
            "dicom_id": row["dicom_id"],
            "study_id": row["study_id"],
            "subject_id": row["subject_id"],
        }


# ==============================================================================
# MIMIC-CXR + MIMIC-IV 14 CONCEPTS & IN-HOSPITAL MORTALITY DATASET (CBM / ECBM)
# ==============================================================================

CHEXPERT_14_CONCEPTS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TOTAL_CONCEPT_STATES = len(CHEXPERT_14_CONCEPTS) * 2  # 28 states (absent=0, present=1)
CONCEPT_OFFSETS = {name: 2 * i for i, name in enumerate(CHEXPERT_14_CONCEPTS)}


class MimicCxrConceptDataset(Dataset):
    """
    Dataset PyTorch nạp ảnh X-quang lồng ngực (MIMIC-CXR), 14 khái niệm CheXpert lâm sàng
    và nhãn kết cục tử vong trong bệnh viện (MIMIC-IV hospital_expire_flag).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str | list[str] | None = None,
        image_size: int = 224,
        training: bool = False,
        transform=None,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        manifest_path = Path(manifest_path).expanduser().resolve()

        import pandas as pd

        self.df = pd.read_csv(manifest_path, dtype={"is_inconsistent_profile": bool})

        if split is not None:
            if isinstance(split, str):
                s_val = "valid" if split.lower() in ("val", "validate", "valid") else split.lower()
                self.df = self.df[self.df["split"] == s_val].copy()
            elif isinstance(split, (list, tuple)):
                s_vals = ["valid" if s.lower() in ("val", "validate", "valid") else s.lower() for s in split]
                self.df = self.df[self.df["split"].isin(s_vals)].copy()

        self.df = self.df.reset_index(drop=True)

        if transform is not None:
            self.transform = transform
        else:
            is_train = training or (split == "train")
            self.transform = image_transforms(image_size, training=is_train)

        self.concept_names = list(CHEXPERT_14_CONCEPTS)
        self.concept_offsets = dict(CONCEPT_OFFSETS)
        self.total_concept_states = TOTAL_CONCEPT_STATES

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_rel = row["relative_path"]
        img_path = self.data_root / img_rel

        with Image.open(img_path) as img:
            image_tensor = self.transform(img.convert("RGB"))

        # Target: hospital_expire_flag (0: Survived, 1: Deceased)
        target_y = int(row["diagnosis_binary"])
        label_tensor = torch.tensor(target_y, dtype=torch.long)

        # 14 Concepts extraction:
        concept_labels = {}
        concept_indices_list = []
        concept_onehot = torch.zeros(self.total_concept_states, dtype=torch.float32)

        for i, c_name in enumerate(self.concept_names):
            state_val = row[c_name]
            c_idx = 1 if state_val == "present" else 0
            concept_labels[c_name] = torch.tensor(c_idx, dtype=torch.long)
            concept_indices_list.append(c_idx)

            # Global index in 28-dimensional one-hot vector:
            global_idx = self.concept_offsets[c_name] + c_idx
            concept_onehot[global_idx] = 1.0

        concept_indices = torch.tensor(concept_indices_list, dtype=torch.long)

        meta = {
            "source_index": int(row["source_index"]),
            "subject_id": str(row["subject_id"]),
            "study_id": str(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "hadm_id": str(row.get("hadm_id", "")),
            "split": str(row["split"]),
            "diagnosis": str(row["diagnosis"]),
            "concept_profile": str(row["concept_profile"]),
            "is_inconsistent": bool(row["is_inconsistent_profile"]),
            "relative_path": str(img_rel),
        }

        return {
            "image": image_tensor,
            "label": label_tensor,
            "concept_labels": concept_labels,
            "concept_indices": concept_indices,
            "concept_onehot": concept_onehot,
            "meta": meta,
        }

