#!/usr/bin/env python3
"""Add auditable, pre-CXR clinical covariates without changing the source cohort.

Run from the project root:
  python3 -B experiments/build_multimodal_clinical.py extract
  python3 -B experiments/build_multimodal_clinical.py build

No imputation, scaling, outcome-guided feature selection, or model fitting occurs.
Only aggregate information is printed. Derived row-level files stay under data/.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/mimic_cxr/dataset/manifest.csv"
DEFAULT_SOURCE = ROOT / "mimic-iv-3.1"
DEFAULT_OUTPUT = ROOT / "data/mimic_cxr/multimodal_v1"
CACHE_VERSION = 1
LOOKBACK_HOURS = 24
LAB_COLUMNS = ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
               "charttime", "storetime", "value", "valuenum", "valueuom"]
SOURCES = {
    "labevents": "https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html",
    "patients": "https://mimic.mit.edu/docs/iv/modules/hosp/patients.html",
    "mimic_iv": "https://physionet.org/content/mimiciv/3.1/",
    "chemistry": "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/chemistry.sql",
    "cbc": "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/complete_blood_count.sql",
}


@dataclass(frozen=True)
class Lab:
    name: str
    itemid: str
    label: str
    category: str
    unit: str
    accepted_units: tuple[str, ...]
    max_value: float | None = None


# IDs are checked against the LOCAL v3.1 dictionary before extraction/build.
# Upper cleaning limits follow the official chemistry SQL, not normal ranges.
LABS = (
    Lab("albumin", "50862", "Albumin", "Chemistry", "g/dL", ("g/dl",), 10),
    Lab("bicarbonate", "50882", "Bicarbonate", "Chemistry", "mEq/L", ("meq/l", "mmol/l"), 10000),
    Lab("chloride", "50902", "Chloride", "Chemistry", "mEq/L", ("meq/l", "mmol/l"), 10000),
    Lab("creatinine", "50912", "Creatinine", "Chemistry", "mg/dL", ("mg/dl",), 150),
    Lab("glucose", "50931", "Glucose", "Chemistry", "mg/dL", ("mg/dl",), 10000),
    Lab("potassium", "50971", "Potassium", "Chemistry", "mEq/L", ("meq/l", "mmol/l"), 30),
    Lab("sodium", "50983", "Sodium", "Chemistry", "mEq/L", ("meq/l", "mmol/l"), 200),
    Lab("bun", "51006", "Urea Nitrogen", "Chemistry", "mg/dL", ("mg/dl",), 300),
    Lab("hematocrit", "51221", "Hematocrit", "Hematology", "%", ("%",), 100),
    Lab("hemoglobin", "51222", "Hemoglobin", "Hematology", "g/dL", ("g/dl",)),
    Lab("platelets", "51265", "Platelet Count", "Hematology", "K/uL", ("k/ul", "10^3/ul", "10^9/l")),
    Lab("wbc", "51301", "White Blood Cells", "Hematology", "K/uL", ("k/ul", "10^3/ul", "10^9/l")),
)
BY_ITEM = {x.itemid: x for x in LABS}
DEMOGRAPHICS = ["clinical_age_years_approx", "clinical_age_is_topcoded", "clinical_sex_recorded"]
VALUE_COLUMNS = DEMOGRAPHICS + ["clinical_" + x.name for x in LABS]
MASK_COLUMNS = [c + "_missing" for c in VALUE_COLUMNS]
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, columns: list[str], rows) -> None:
    # Atomic publication; incomplete files never masquerade as finished artifacts.
    tmp = path.with_name(path.name + ".partial")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_timestamp(value: str) -> datetime | None:
    if not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.strip())
        return result if result.tzinfo is None else None
    except ValueError:
        return None


def study_timestamp(date: str, clock: str) -> datetime:
    require(bool(re.fullmatch(r"\d{8}", date)), "Invalid CXR StudyDate; no fallback is allowed")
    match = re.fullmatch(r"(\d{1,6})(?:\.(\d+))?", clock)
    require(match is not None, "Invalid CXR StudyTime; no midnight fallback is allowed")
    digits = match.group(1).zfill(6)
    # Published CXR CSVs contain long floating-point decimal expansions. Floor
    # sub-microsecond digits so the cutoff can never move to a later instant.
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    return datetime.strptime(date + digits + fraction, "%Y%m%d%H%M%S%f")


def iso(value: datetime) -> str:
    return value.isoformat(sep=" ")


def normalized_unit(value: str) -> str:
    return value.strip().casefold().replace("µ", "u").replace("μ", "u").replace(" ", "")


def numeric_result(event: dict[str, str], lab: Lab) -> tuple[str, str]:
    """Conservative conversion: never replace censored or textual values with a number."""
    raw, number = event["value"].strip(), event["valuenum"].strip()
    if any(x in raw for x in ("<", ">", "≤", "≥")):
        return "", "censored_result"
    redacted_or_empty_text = raw in {"", "___"}
    if not NUMERIC_PATTERN.fullmatch(number) or (not redacted_or_empty_text and not NUMERIC_PATTERN.fullmatch(raw)):
        return "", "non_numeric_result"
    try:
        value = Decimal(number)
        textual = Decimal(raw) if not redacted_or_empty_text else value
    except InvalidOperation:
        return "", "non_numeric_result"
    if not value.is_finite() or not textual.is_finite():
        return "", "non_finite_result"
    tolerance = max(Decimal("0.000001"), abs(value) * Decimal("0.000001"))
    if abs(value - textual) > tolerance:
        return "", "value_valuenum_disagree"
    if normalized_unit(event["valueuom"]) not in lab.accepted_units:
        return "", "unrecognized_or_missing_unit"
    if value <= 0:
        return "", "nonpositive_result"
    if lab.max_value is not None and value > Decimal(str(lab.max_value)):
        return "", "above_cleaning_limit"
    return format(value.normalize(), "f"), "observed"


def event_time_status(event: dict[str, str], context: dict) -> tuple[str, tuple | None]:
    if event["subject_id"] != context["subject_id"]:
        return "wrong_subject", None
    if not event["hadm_id"]:
        return "missing_hadm_id", None
    if event["hadm_id"] != context["hadm_id"]:
        return "other_admission", None
    chart, store = parse_timestamp(event["charttime"]), parse_timestamp(event["storetime"])
    if chart is None or store is None:
        return "missing_or_invalid_lab_timestamp", None
    if store < chart:
        return "storetime_before_charttime", None
    if chart < context["window_start"]:
        return "before_window", None
    if chart > context["t0"]:
        return "sample_after_prediction", None
    if store > context["t0"]:
        return "result_after_prediction", None
    return "eligible", (chart, store)


def age_features(patient: dict[str, str], t0: datetime) -> tuple[str, str]:
    anchor_age, anchor_year = int(patient["anchor_age"]), int(patient["anchor_year"])
    require(0 <= anchor_age <= 91, "Invalid anchor_age")
    approximate = anchor_age + t0.year - anchor_year
    require(0 <= approximate <= 120, "Invalid derived age")
    # 91 is an explicit open-ended category, NEVER an exact age. Cap new >89
    # estimates too; do not fabricate 92/98/etc from the deidentified 91 sentinel.
    topcoded = anchor_age == 91 or approximate > 89
    return ("91" if topcoded else str(approximate)), str(int(topcoded))


def source_inventory(source: Path, manifest: Path) -> dict:
    expected = {}
    with (source / "SHA256SUMS.txt").open() as stream:
        for line in stream:
            digest, filename = line.strip().split(maxsplit=1)
            expected[filename.lstrip("*")] = digest
    inventory = {}
    for filename in ("hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/d_labitems.csv.gz", "hosp/labevents.csv.gz"):
        path = source / filename
        digest = sha256(path)
        require(digest == expected.get(filename), "Source checksum mismatch: " + filename)
        inventory[filename] = {"sha256": digest, "bytes": path.stat().st_size,
                               "matches_distributed_checksum": True}
    inventory["input_manifest"] = {"sha256": sha256(manifest), "bytes": manifest.stat().st_size}
    return inventory


def check_dictionary(source: Path) -> dict:
    _, rows = read_csv(source / "hosp/d_labitems.csv.gz")
    dictionary = {r["itemid"]: r for r in rows}
    for lab in LABS:
        entry = dictionary[lab.itemid]
        require((entry["label"], entry["fluid"], entry["category"]) ==
                (lab.label, "Blood", lab.category), "Lab dictionary mismatch: " + lab.name)
    return {lab.itemid: dictionary[lab.itemid] for lab in LABS}


def load_context(manifest: Path, source: Path) -> tuple[list[str], list[dict], dict, list[dict], dict]:
    columns, rows = read_csv(manifest)
    require(rows and len({r["subject_id"] for r in rows}) == len(rows), "Expected one row per patient")
    for key in ("hadm_id", "dicom_id"):
        require(len({r[key] for r in rows}) == len(rows), "Duplicate cohort " + key)
    require({r["split"] for r in rows} <= {"train", "valid", "test"}, "Unknown split")
    subjects = {r["subject_id"] for r in rows}
    with gzip.open(source / "hosp/patients.csv.gz", "rt", newline="") as stream:
        patients = {r["subject_id"]: r for r in csv.DictReader(stream) if r["subject_id"] in subjects}
    admissions, by_subject = {}, defaultdict(list)
    with gzip.open(source / "hosp/admissions.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["subject_id"] in subjects:
                require(row["hadm_id"] not in admissions, "Duplicate admission key")
                admissions[row["hadm_id"]] = row
                by_subject[row["subject_id"]].append(row)
    contexts, exclusions, audit = {}, [], Counter()
    for row in rows:
        subject, hadm = row["subject_id"], row["hadm_id"]
        require(subject in patients and hadm in admissions, "Missing patient/admission linkage")
        admission = admissions[hadm]
        require(admission["subject_id"] == subject, "Patient/admission linkage mismatch")
        require(admission["hospital_expire_flag"] == row["diagnosis_binary"], "Outcome disagrees with source")
        require(admission["admission_type"] == row["admission_type"], "Admission metadata disagrees")
        t0 = study_timestamp(row["StudyDate"], row["StudyTime"])
        admit, discharge = parse_timestamp(admission["admittime"]), parse_timestamp(admission["dischtime"])
        ed = parse_timestamp(admission["edregtime"])
        require(admit is not None and discharge is not None, "Invalid encounter timestamps")
        start = min(admit, ed) if ed is not None else admit
        require(start <= t0 <= discharge, "CXR falls outside linked encounter")
        death = parse_timestamp(admission["deathtime"])
        require((row["diagnosis_binary"] == "1") == (death is not None), "Outcome/death timestamp mismatch")
        overlaps = 0
        for other in by_subject[subject]:
            oa, od = parse_timestamp(other["admittime"]), parse_timestamp(other["dischtime"])
            oe = parse_timestamp(other["edregtime"])
            if oa is not None and od is not None and (min(oa, oe) if oe else oa) <= t0 <= od:
                overlaps += 1
        reasons = []
        if death is not None and t0 >= death:
            reasons.append("prediction_not_before_recorded_death")
        if overlaps != 1:
            reasons.append("ambiguous_overlapping_encounters")
        for reason in reasons:
            audit[reason] += 1
        context = {"subject_id": subject, "hadm_id": hadm, "dicom_id": row["dicom_id"],
                   "t0": t0, "window_start": max(start, t0 - timedelta(hours=LOOKBACK_HOURS)),
                   "encounter_start": start, "admittime": admit, "eligible": not reasons,
                   "exclusion_reason": "|".join(reasons), "patient": patients[subject]}
        contexts[subject] = context
        audit["cxr_times_with_submicrosecond_digits_floored"] += int("." in row["StudyTime"] and len(row["StudyTime"].split(".")[1]) > 6)
        audit["images_before_formal_admission"] += int(t0 < admit)
        audit["images_over_24h_after_admission"] += int(t0 > admit + timedelta(hours=24))
        if reasons:
            exclusions.append({**{k: row[k] for k in ("dicom_id", "subject_id", "hadm_id", "split", "diagnosis_binary")},
                               "prediction_time": iso(t0), "recorded_deathtime": admission["deathtime"],
                               "reason": "|".join(reasons)})
    return columns, rows, contexts, exclusions, dict(audit)


def cache_signature(inventory: dict, manifest: Path) -> dict:
    return {"cache_version": CACHE_VERSION,
            "manifest_sha256": inventory["input_manifest"]["sha256"],
            "source_sha256": inventory["hosp/labevents.csv.gz"]["sha256"],
            "itemids": sorted(BY_ITEM)}


def extract(args) -> None:
    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=True)
    print("Checking compressed source checksums against MIMIC-IV SHA256SUMS.txt", flush=True)
    inventory = source_inventory(args.source, args.manifest)
    check_dictionary(args.source)
    _, cohort = read_csv(args.manifest)
    subjects = {r["subject_id"] for r in cohort}
    hadms = {r["hadm_id"] for r in cohort}
    signature = cache_signature(inventory, args.manifest)
    cache = args.output / "cohort_lab_events.csv.gz"
    metadata = args.output / "cohort_lab_events.meta.json"
    if cache.exists() and metadata.exists():
        meta = json.loads(metadata.read_text())
        require(meta["signature"] == signature, "Existing cache has different inputs; use a new output directory")
        require(meta["cache_sha256"] == sha256(cache), "Cached events checksum mismatch")
        print("Verified existing extraction cache; no rescan needed", flush=True)
        return
    tmp = cache.with_name(cache.name + ".partial")
    total = kept = 0
    started = time.monotonic()
    with gzip.open(tmp, "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        for index, frame in enumerate(pd.read_csv(args.source / "hosp/labevents.csv.gz", usecols=LAB_COLUMNS,
                                                  dtype=str, keep_default_na=False, chunksize=args.chunk_size)):
            total += len(frame)
            mask = frame["itemid"].isin(BY_ITEM) & (frame["subject_id"].isin(subjects) | frame["hadm_id"].isin(hadms))
            selected = frame.loc[mask, LAB_COLUMNS]
            kept += len(selected)
            selected.to_csv(stream, index=False, header=index == 0)
            if index % 10 == 0:
                print(json.dumps({"source_rows_scanned": total, "retained_rows": kept,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    require(total > 0, "Empty labevents source")
    tmp.replace(cache)
    write_json(metadata, {"signature": signature, "source_inventory": inventory,
                          "source_rows_scanned": total, "retained_rows": kept,
                          "cache_sha256": sha256(cache)})
    print(json.dumps({"extraction_complete": True, "source_rows_scanned": total, "retained_rows": kept}), flush=True)


def build(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "validation_report.json", {"passed": False, "status": "pending_after_build"})
    inventory = source_inventory(args.source, args.manifest)
    dictionary = check_dictionary(args.source)
    cache = args.output / "cohort_lab_events.csv.gz"
    meta = json.loads((args.output / "cohort_lab_events.meta.json").read_text())
    require(meta["signature"] == cache_signature(inventory, args.manifest), "Cache belongs to different inputs")
    require(meta["cache_sha256"] == sha256(cache), "Cached events checksum mismatch")
    original_columns, cohort, contexts, exclusions, context_audit = load_context(args.manifest, args.source)
    # Verify the base cohort hash recorded by the prior build, where available.
    source_report = args.manifest.parent / "report.json"
    if source_report.exists():
        require(json.loads(source_report.read_text())["manifest_sha256"] == sha256(args.manifest),
                "Base cohort report/manifest checksum mismatch")
    # Strict metadata round-trip against the full original CXR metadata.
    cxr_metadata = args.manifest.parent.parent / "mimic-cxr-2.0.0-metadata.csv.gz"
    require(cxr_metadata.exists(), "Original CXR metadata is required")
    cohort_by_image = {r["dicom_id"]: r for r in cohort}
    verified_images = set()
    with gzip.open(cxr_metadata, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dicom_id"] in cohort_by_image:
                require(row["dicom_id"] not in verified_images, "Duplicate original CXR metadata key")
                base = cohort_by_image[row["dicom_id"]]
                for column in ("subject_id", "study_id", "StudyDate", "StudyTime", "ViewPosition"):
                    require(row[column] == base[column], "CXR metadata mismatch: " + column)
                verified_images.add(row["dicom_id"])
    require(len(verified_images) == len(cohort), "Some cohort images lack source metadata")
    require(all((ROOT / r["image_path"]).is_file() for r in cohort), "Some cohort images are missing")
    inventory["cxr_metadata"] = {"sha256": sha256(cxr_metadata), "bytes": cxr_metadata.stat().st_size}

    event_counts, per_key_counts = Counter(), defaultdict(Counter)
    units = defaultdict(Counter)
    latest = {}
    seen_ids = set()
    with gzip.open(cache, "rt", newline="") as stream:
        for event in csv.DictReader(stream):
            require(event["labevent_id"] not in seen_ids, "Duplicate source lab event ID in cache")
            seen_ids.add(event["labevent_id"])
            context = contexts.get(event["subject_id"])
            # A foreign subject attached to a cohort admission is a hard linkage error.
            require(context is not None, "Source lab admission belongs to a different subject")
            lab = BY_ITEM[event["itemid"]]
            key = (event["subject_id"], lab.name)
            status, rank = event_time_status(event, context)
            event_counts[status] += 1
            per_key_counts[key][status] += 1
            units[lab.name][event["valueuom"] or "<EMPTY>"] += 1
            if status != "eligible":
                continue
            value, quality = numeric_result(event, lab)
            candidate = {"rank": rank, "event": event, "value": value, "quality": quality,
                         "signatures": {(value, quality)}, "ties": 1}
            if key not in latest or rank > latest[key]["rank"]:
                latest[key] = candidate
            elif rank == latest[key]["rank"]:
                previous = latest[key]
                previous["ties"] += 1
                previous["signatures"].add((value, quality))
                if int(event["labevent_id"]) < int(previous["event"]["labevent_id"]):
                    previous["event"] = event
    require(len(seen_ids) == meta["retained_rows"], "Cache row count mismatch")
    augmented, provenance, demographic_provenance = [], [], []
    lab_status_counts = defaultdict(Counter)
    for base in cohort:
        context = contexts[base["subject_id"]]
        output = dict(base)
        output.update({"prediction_time": iso(context["t0"]), "clinical_window_start": iso(context["window_start"]),
                       "prediction_eligible": str(int(context["eligible"])),
                       "prediction_exclusion_reason": context["exclusion_reason"]})
        age, topcoded = age_features(context["patient"], context["t0"])
        sex = context["patient"]["gender"].strip()
        require(sex in {"F", "M"}, "Unexpected recorded sex; explicit handling is required")
        output.update(dict(zip(DEMOGRAPHICS, (age, topcoded, sex))))
        for column in DEMOGRAPHICS:
            output[column + "_missing"] = "0"
        demographic_provenance.append({"dicom_id": base["dicom_id"], "subject_id": base["subject_id"],
                                       "hadm_id": base["hadm_id"], "prediction_time": iso(context["t0"]),
                                       "anchor_age": context["patient"]["anchor_age"],
                                       "anchor_year": context["patient"]["anchor_year"],
                                       "source_gender": sex, "age_years_approx": age,
                                       "age_is_topcoded": topcoded})
        for lab in LABS:
            key = (base["subject_id"], lab.name)
            candidate = latest.get(key)
            event = candidate["event"] if candidate else {column: "" for column in LAB_COLUMNS}
            status = candidate["quality"] if candidate else "no_eligible_timed_event"
            if candidate and len(candidate["signatures"]) > 1:
                status = "conflicting_latest_events"
            value = candidate["value"] if candidate and status == "observed" else ""
            name = "clinical_" + lab.name
            output[name], output[name + "_missing"] = value, str(int(value == ""))
            if context["eligible"]:
                lab_status_counts[lab.name][status] += 1
            provenance.append({"dicom_id": base["dicom_id"], "subject_id": base["subject_id"],
                               "hadm_id": base["hadm_id"], "feature": name, "itemid": lab.itemid,
                               "canonical_unit": lab.unit, "status": status, "normalized_value": value,
                               "prediction_time": iso(context["t0"]), "window_start": iso(context["window_start"]),
                               "source_labevent_id": event["labevent_id"], "source_subject_id": event["subject_id"],
                               "source_hadm_id": event["hadm_id"], "specimen_id": event["specimen_id"],
                               "charttime": event["charttime"], "storetime": event["storetime"],
                               "raw_value": event["value"], "raw_valuenum": event["valuenum"],
                               "raw_unit": event["valueuom"],
                               "raw_text_missing_or_redacted": str(int(event["value"].strip() in {"", "___"})),
                               "tied_latest_events": candidate["ties"] if candidate else 0,
                               "eligible_event_count": per_key_counts[key]["eligible"]})
        augmented.append(output)
    usable = [row for row in augmented if row["prediction_eligible"] == "1"]
    require(usable, "No temporally eligible patients remain")
    new_columns = ["prediction_time", "clinical_window_start", "prediction_eligible", "prediction_exclusion_reason"] + VALUE_COLUMNS + MASK_COLUMNS
    require(not set(new_columns) & set(original_columns), "Clinical columns already exist in base manifest")
    full_columns = original_columns + new_columns
    write_csv(args.output / "manifest_all.csv", full_columns, augmented)
    write_csv(args.output / "manifest.csv", full_columns, usable)
    write_csv(args.output / "clinical_features.csv", ["dicom_id", "subject_id", "hadm_id", "split"] + VALUE_COLUMNS + MASK_COLUMNS,
              ({key: row[key] for key in ["dicom_id", "subject_id", "hadm_id", "split"] + VALUE_COLUMNS + MASK_COLUMNS} for row in usable))
    write_csv(args.output / "lab_provenance.csv.gz", list(provenance[0]), provenance)
    write_csv(args.output / "demographic_provenance.csv", list(demographic_provenance[0]), demographic_provenance)
    exclusion_columns = ["dicom_id", "subject_id", "hadm_id", "split", "diagnosis_binary", "prediction_time", "recorded_deathtime", "reason"]
    write_csv(args.output / "excluded_rows.csv", exclusion_columns, exclusions)
    coverage = []
    for split in ("all", "train", "valid", "test"):
        subset = usable if split == "all" else [r for r in usable if r["split"] == split]
        for column in VALUE_COLUMNS:
            values = [r[column] for r in subset if r[column] != ""]
            numeric = [float(value) for value in values] if column != "clinical_sex_recorded" else []
            coverage.append({"split": split, "feature": column, "rows": len(subset), "observed": len(values),
                             "missing": len(subset) - len(values), "observed_pct": round(100 * len(values) / len(subset), 4),
                             "min": min(numeric) if numeric else "", "max": max(numeric) if numeric else ""})
    write_csv(args.output / "coverage.csv", list(coverage[0]), coverage)
    schema = {"schema_version": 1, "clinical_value_columns": VALUE_COLUMNS, "clinical_missing_columns": MASK_COLUMNS,
              "model_clinical_columns": VALUE_COLUMNS + MASK_COLUMNS,
              "numeric_value_columns": [x for x in VALUE_COLUMNS if x != "clinical_sex_recorded"],
              "categorical_value_columns": ["clinical_sex_recorded"], "categorical_values": {"clinical_sex_recorded": ["F", "M"]},
              "concept_label_columns": [c.removeprefix("raw_") for c in original_columns if c.startswith("raw_")],
              "outcome_column": "diagnosis_binary", "split_column": "split", "image_path_column": "image_path",
              "feature_allowlist_required": True,
              "metadata_is_not_model_input": [c for c in full_columns if c not in VALUE_COLUMNS + MASK_COLUMNS],
              "missing_policy": "Empty numeric cells remain missing; *_missing=1. Fit any imputation/scaling on train only.",
              "age_policy": "Approximate calendar-year age; anchor_age=91 or estimated age>89 is encoded as open-ended 91 plus topcoded=1; not exact age.",
              "lab_definitions": [asdict(lab) for lab in LABS]}
    write_json(args.output / "feature_schema.json", schema)
    by_split = {s: {"rows": sum(r["split"] == s for r in usable),
                    "deaths": sum(int(r["diagnosis_binary"]) for r in usable if r["split"] == s)} for s in ("train", "valid", "test")}
    report = {"protocol_version": 1, "input_rows": len(cohort), "eligible_rows": len(usable), "excluded_rows": len(exclusions),
              "by_split": by_split, "lookback_hours": LOOKBACK_HOURS,
              "prediction_time": "Original CXR StudyDate + StudyTime; fractional seconds retained to microseconds, extra digits floored (cutoff never moved later)",
              "window": "[max(min(edregtime, admittime), prediction_time - 24 hours), prediction_time]",
              "lab_linkage": "Exact subject_id AND exact nonempty hadm_id; no inferred admission IDs",
              "lab_availability": "window_start <= charttime <= storetime <= prediction_time",
              "lab_selection": "Latest eligible (charttime, storetime); conflicting tied results are missing; no fallback to an older numeric result if latest is invalid",
              "numeric_source": "Use numeric valuenum with a recognized unit; compare numeric value text when present. Blank/___ value text does not invalidate an available valuenum; censored/other textual results are not converted.",
              "time_context_audit": context_audit, "source_inventory": inventory, "source_dictionary": dictionary,
              "lab_event_filter_counts": dict(event_counts),
              "lab_status_counts_eligible_cohort": {k: dict(v) for k, v in lab_status_counts.items()},
              "source_unit_inventory_all_cached_events": {k: dict(v) for k, v in units.items()},
              "patients_with_any_lab": sum(any(r["clinical_" + lab.name] != "" for lab in LABS) for r in usable),
              "patients_with_no_lab": sum(all(r["clinical_" + lab.name] == "" for lab in LABS) for r in usable),
              "base_cohort_and_existing_splits_preserved": True, "imputation_or_scaling_performed": False,
              "clinical_feature_count_without_masks": len(VALUE_COLUMNS), "clinical_missing_mask_count": len(MASK_COLUMNS),
              "source_references": SOURCES, "builder_sha256": sha256(Path(__file__)),
              "runtime_versions": {"python": sys.version.split()[0], "pandas": importlib.metadata.version("pandas")},
              "limitations": ["Retrospective source records can contain errors; timestamp checks do not certify real-world clinical truth.",
                              "Age is approximate and older ages are topcoded.",
                              "Strict admission linkage excludes unassigned labs, including some ED/outpatient observations.",
                              "Only 12 prespecified blood lab IDs; no ICU vitals, blood gases, text, or discharge-finalized predictors.",
                              "Mortality was enriched in the base cohort; probabilities require population-appropriate evaluation.",
                              "Source clinical cleaning limits are not a guarantee that every surviving measurement is physiologically correct."]}
    report["output_sha256"] = {path.name: sha256(path) for path in sorted(args.output.iterdir())
                               if path.name in {"manifest.csv", "manifest_all.csv", "clinical_features.csv", "lab_provenance.csv.gz",
                                                "demographic_provenance.csv", "excluded_rows.csv", "coverage.csv", "feature_schema.json"}}
    write_json(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("input_rows", "eligible_rows", "excluded_rows", "by_split", "patients_with_any_lab", "patients_with_no_lab")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "build"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    require(args.chunk_size > 0, "Chunk size must be positive")
    (extract if args.stage == "extract" else build)(args)


if __name__ == "__main__":
    main()
