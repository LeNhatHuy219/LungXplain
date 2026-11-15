"""Dataset v5 final build (protocol: experiments/DATASET_V5.md incl. the 15/11/2026 build addendum).

Reads the frozen selection in data/mimic_cxr/cohort_v5r2/ and writes
data/mimic_cxr/multimodal_v5/: every image re-verified (published SHA-256 and a full
JPEG decode), demographics, the selected labs, MIMIC-IV-ED triage vitals, concept
labels in v3 form (U-Zeros + known/ignore) and v4 form (nullable + mask/state), a
separate radiologist-label file, an exact-hadm_id-only sensitivity manifest, the
feature schema, provenance and an output-hash report. The builder never writes
validation_report.json; only the independent validator does. The selection script
is not modified. Patient-level files stay under the ignored data/ directory; stdout
is aggregate only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multimodal_clinical as clinical  # noqa: E402
from download_images import load_checksums  # noqa: E402
from prepare_mimic_mortality_cohort import CHEXPERT_14_CONCEPTS  # noqa: E402

ROOT = clinical.ROOT
DATA = ROOT / "data/mimic_cxr"
COHORT = DATA / "cohort_v5r2"
OUTPUT = DATA / "multimodal_v5"
SOURCE = ROOT / "mimic-iv-3.1"
ED = ROOT / "mimic-iv-ed-2.2"
PROTOCOL = ROOT / "experiments/DATASET_V5.md"
SELECTOR = ROOT / "experiments/build_dataset_v5.py"
LAB_NAMES = tuple(lab.name for lab in clinical.LABS)
CORE = ("bicarbonate", "chloride", "creatinine", "glucose", "potassium", "sodium", "bun",
        "hematocrit", "hemoglobin", "platelets", "wbc")
# Inclusive plausibility ranges fixed in the protocol before any ED value was on disk.
TRIAGE = {"temperature": (Decimal(86), Decimal(113), "degF"), "heartrate": (Decimal(20), Decimal(300), "beats/min"),
          "resprate": (Decimal(3), Decimal(80), "breaths/min"), "o2sat": (Decimal(50), Decimal(100), "%"),
          "sbp": (Decimal(40), Decimal(300), "mmHg"), "dbp": (Decimal(10), Decimal(200), "mmHg"),
          "acuity": (Decimal(1), Decimal(5), "ESI level (integer)")}
DEMOGRAPHICS = ["clinical_age_years_approx", "clinical_age_is_topcoded", "clinical_sex_recorded"]
LAB_COLUMNS = ["clinical_" + name for name in LAB_NAMES]
TRIAGE_COLUMNS = ["clinical_triage_" + name for name in TRIAGE]
VALUE_COLUMNS = DEMOGRAPHICS + LAB_COLUMNS + TRIAGE_COLUMNS
MISSING_COLUMNS = [name + "_missing" for name in VALUE_COLUMNS]
STATES = {"1": "positive", "0": "explicit_negative", "-1": "uncertain"}
EXPECTED = {"rows": 5000, "splits": {"train": 4000, "valid": 500, "test": 500},
            "deaths": {"train": 400, "valid": 50, "test": 50}}
SHORT_TRIAGE_MINUTES = 30


def distributed_sums(path):
    sums = {}
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip().lstrip("*")] = parts[0]
    return sums


def verify_sources():
    inventory = {}
    checks = [(SOURCE, distributed_sums(SOURCE / "SHA256SUMS.txt"), ("hosp/patients.csv.gz", "hosp/admissions.csv.gz")),
              (ED, distributed_sums(ED / "SHA256SUMS.txt"), ("ed/edstays.csv.gz", "ed/triage.csv.gz")),
              (DATA, load_checksums(), ("mimic-cxr-2.0.0-chexpert.csv.gz", "mimic-cxr-2.0.0-metadata.csv.gz",
                                        "mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.1.0-test-set-labeled.csv"))]
    for root, sums, names in checks:
        for name in names:
            digest = clinical.sha256(root / name)
            clinical.require(digest == sums.get(name), "Source checksum mismatch: " + name)
            inventory[f"{root.name}/{name}"] = {"sha256": digest, "matches_distributed_checksum": True}
    return inventory


def load_selection(cohort_dir):
    frame = pd.read_csv(cohort_dir / "manifest.csv", dtype=str, keep_default_na=False)
    selection = json.loads((cohort_dir / "selection_report.json").read_text())
    clinical.require(selection["selector_sha256"] == clinical.sha256(SELECTOR), "Selection script changed after selection")
    clinical.require(len(frame) == EXPECTED["rows"] and selection["selected_rows"] == EXPECTED["rows"], "Selection is not 5000 rows")
    for key in ("subject_id", "hadm_id", "dicom_id", "study_id"):
        clinical.require(frame[key].nunique() == EXPECTED["rows"], "Duplicate " + key)
    clinical.require(frame["split"].value_counts().to_dict() == EXPECTED["splits"], "Unexpected split sizes")
    deaths = frame.groupby("split")["diagnosis_binary"].apply(lambda s: int((s == "1").sum())).to_dict()
    clinical.require(deaths == EXPECTED["deaths"], "Unexpected deaths by split")
    clinical.require(set(frame["diagnosis_binary"]) <= {"0", "1"}, "Invalid outcome")
    for lab in CORE:
        clinical.require((frame["clinical_" + lab] != "").all(), "A required lab is missing in the selection")
    labeled = frame["radiologist_labeled"] == "1"
    clinical.require((frame.loc[labeled, "split"] == "test").all(), "A radiologist-labeled study is outside test")
    return frame, selection


def audit_image(record):
    path = ROOT / record["image_path"]
    clinical.require(path.resolve().is_relative_to(DATA.resolve()), "Image outside data/mimic_cxr")
    digest = clinical.sha256(path)
    with Image.open(path) as image:
        image.load()
        clinical.require(image.format == "JPEG", "Source image is not JPEG")
        width, height, mode = image.size[0], image.size[1], image.mode
    return {"dicom_id": record["dicom_id"], "image_path": record["image_path"], "sha256": digest,
            "published_sha256": record["published_sha256"], "bytes": path.stat().st_size,
            "width": width, "height": height, "mode": mode, "decoded": 1}


def audit_images(frame, workers):
    sums = load_checksums()
    records = []
    for row in frame.itertuples():
        clinical.require(row.relative_path in sums, "Image lacks a published checksum")
        clinical.require((ROOT / row.image_path).is_file(), "A selected image is missing; download it first")
        records.append({"dicom_id": row.dicom_id, "image_path": row.image_path, "published_sha256": sums[row.relative_path]})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        audit = pd.DataFrame(list(pool.map(audit_image, records)))
    clinical.require((audit["sha256"] == audit["published_sha256"]).all(), "Image checksum mismatch")
    clinical.require(audit["sha256"].is_unique, "Duplicate image content")
    return audit


def demographics(frame):
    patients = pd.read_csv(SOURCE / "hosp/patients.csv.gz", dtype=str, keep_default_na=False).set_index("subject_id")
    rows = []
    for row in frame.itertuples():
        patient = patients.loc[row.subject_id].to_dict()
        t0 = pd.Timestamp(row.prediction_time).to_pydatetime()
        age, topcoded = clinical.age_features(patient, t0)
        clinical.require(patient["gender"] in {"F", "M"}, "Unknown sex")
        rows.append({"dicom_id": row.dicom_id, "subject_id": row.subject_id, "hadm_id": row.hadm_id,
                     "prediction_time": row.prediction_time, "anchor_age": patient["anchor_age"],
                     "anchor_year": patient["anchor_year"], "source_gender": patient["gender"],
                     "age_years_approx": age, "age_is_topcoded": topcoded})
    return pd.DataFrame(rows)


def clean_triage(name, raw):
    """Return (normalized value or '', status) under the protocol ranges."""
    text = raw.strip()
    if text == "":
        return "", "missing_in_source"
    if not clinical.NUMERIC_PATTERN.fullmatch(text):
        return "", "non_numeric"
    try:
        value = Decimal(text)
    except InvalidOperation:
        return "", "non_numeric"
    low, high, _ = TRIAGE[name]
    if not value.is_finite() or value < low or value > high:
        return "", "out_of_range"
    if name == "acuity":
        if value != value.to_integral_value():
            return "", "non_integer_acuity"
        return str(int(value)), "observed"
    return format(value.normalize(), "f"), "observed"


def link_triage(context, stays, triage):
    """context: dicom_id, subject_id, hadm_id, encounter_start, prediction_time (str).

    Returns one row per context row with cleaned triage values, statuses and the
    linked stay. Pure function of its inputs (unit-tested with synthetic tables)."""
    clinical.require(triage["stay_id"].is_unique, "triage.stay_id is not unique")
    clinical.require(stays["stay_id"].is_unique, "edstays.stay_id is not unique")
    ctx = context.copy()
    ctx["start_ts"] = pd.to_datetime(ctx["encounter_start"], format="ISO8601")
    ctx["t0_ts"] = pd.to_datetime(ctx["prediction_time"], format="ISO8601")
    candidate = stays[stays["hadm_id"].isin(set(ctx["hadm_id"]))].rename(columns={"subject_id": "ed_subject_id"})
    candidate = candidate.merge(ctx[["hadm_id", "subject_id", "start_ts", "t0_ts"]], on="hadm_id")
    clinical.require((candidate["ed_subject_id"] == candidate["subject_id"]).all(), "ED stay hadm_id/subject_id mismatch")
    candidate = candidate.assign(intime_ts=pd.to_datetime(candidate["intime"].replace("", None), format="ISO8601"))
    any_stay = set(candidate["hadm_id"])
    inside = candidate[candidate["intime_ts"].notna() & (candidate["intime_ts"] >= candidate["start_ts"])
                       & (candidate["intime_ts"] <= candidate["t0_ts"])]
    latest = inside[inside["intime_ts"] == inside.groupby("hadm_id")["intime_ts"].transform("max")]
    tie_count = latest.groupby("hadm_id")["stay_id"].transform("size")
    ambiguous = set(latest.loc[tie_count > 1, "hadm_id"])
    chosen = latest[tie_count == 1].set_index("hadm_id")
    tri = triage.set_index("stay_id")
    rows = []
    for row in ctx.itertuples():
        record = {"dicom_id": row.dicom_id, "ed_stay_id": "", "ed_intime": "", "minutes_intime_to_t0": "",
                  "link_status": "linked"}
        raw = {}
        if row.hadm_id in ambiguous:
            record["link_status"] = "ambiguous_latest_ed_stay"
        elif row.hadm_id in chosen.index:
            stay = chosen.loc[row.hadm_id]
            record.update(ed_stay_id=stay["stay_id"], ed_intime=stay["intime"],
                          minutes_intime_to_t0=round((row.t0_ts - stay["intime_ts"]).total_seconds() / 60, 1))
            if stay["stay_id"] in tri.index:
                source = tri.loc[stay["stay_id"]]
                clinical.require(source["subject_id"] == row.subject_id, "triage subject_id mismatch")
                raw = {name: source[name] for name in TRIAGE}
            else:
                record["link_status"] = "no_triage_row"
        else:
            record["link_status"] = "ed_stay_outside_window" if row.hadm_id in any_stay else "no_linked_ed_stay"
        for name in TRIAGE:
            value, status = clean_triage(name, raw[name]) if raw else ("", record["link_status"])
            record["clinical_triage_" + name], record["status_" + name] = value, status
        if record["status_sbp"] == "observed" and record["status_dbp"] == "observed" \
                and Decimal(record["clinical_triage_dbp"]) >= Decimal(record["clinical_triage_sbp"]):
            for name in ("sbp", "dbp"):
                record["clinical_triage_" + name], record["status_" + name] = "", "diastolic_not_below_systolic"
        rows.append(record)
    return pd.DataFrame(rows)


def concept_columns(frame):
    labels = pd.read_csv(DATA / "mimic-cxr-2.0.0-chexpert.csv.gz", dtype=str, keep_default_na=False)
    labels = labels.set_index(["subject_id", "study_id"])
    clinical.require(labels.index.is_unique, "Duplicate CheXpert source row")
    source = labels.loc[list(zip(frame["subject_id"], frame["study_id"]))].reset_index(drop=True)
    out = {}
    for name in CHEXPERT_14_CONCEPTS:
        raw = source[name].astype(str)
        parsed = raw.map(lambda v: "" if v == "" else str(int(Decimal(v))) if Decimal(v) in (-1, 0, 1) else "invalid")
        clinical.require(not (parsed == "invalid").any(), "Invalid concept source state")
        out[name] = (parsed == "1").astype(int).astype(str)
        out["raw_" + name] = raw
        out["ignore_" + name] = (parsed == "-1").astype(int).astype(str)
        out["known_" + name] = parsed.isin(["0", "1"]).astype(int).astype(str)
        out["nullable_" + name] = parsed.where(parsed.isin(["0", "1"]), "")
        out["label_mask_" + name] = out["known_" + name]
        out["state_" + name] = parsed.map(STATES).fillna("not_mentioned")
    result = pd.DataFrame(out)
    counted = [n for n in CHEXPERT_14_CONCEPTS if n not in {"No Finding", "Support Devices"}]
    result["n_findings"] = result[counted].astype(int).sum(axis=1).astype(str)
    return result


def radiologist_labels(frame):
    source = pd.read_csv(DATA / "mimic-cxr-2.1.0-test-set-labeled.csv", dtype=str, keep_default_na=False)
    clinical.require(source["study_id"].is_unique, "Duplicate radiologist-labeled study")
    source = source.rename(columns={"Airspace Opacity": "Lung Opacity"})
    clinical.require(set(CHEXPERT_14_CONCEPTS) <= set(source.columns), "Radiologist label columns changed")
    labeled = frame[frame["radiologist_labeled"] == "1"][["dicom_id", "study_id", "split"]]
    merged = labeled.merge(source[["study_id"] + list(CHEXPERT_14_CONCEPTS)], on="study_id", validate="one_to_one")
    clinical.require(len(merged) == len(labeled) and (merged["split"] == "test").all(), "Radiologist labels incomplete")
    return merged


def missingness(manifest, statuses):
    rows = []
    for split, part in [("all", manifest)] + list(manifest.groupby("split")):
        for column in VALUE_COLUMNS:
            reasons = statuses[column].loc[part.index] if column in statuses else \
                np.where(part[column] == "", "missing", "observed")
            for reason, count in Counter(reasons).items():
                rows.append({"split": split, "feature": column, "protocol_reason": reason, "rows": int(count)})
    return pd.DataFrame(rows).sort_values(["split", "feature", "protocol_reason"])


def feature_schema(concepts):
    numeric = [c for c in VALUE_COLUMNS if c != "clinical_sex_recorded"]
    model_columns = VALUE_COLUMNS + MISSING_COLUMNS
    metadata = [c for c in BASE_COLUMNS + list(concepts.columns) + EXTRA_COLUMNS if c not in model_columns]
    return {"schema_version": 3, "dataset_version": "v5",
            "clinical_value_columns": VALUE_COLUMNS, "clinical_missing_columns": MISSING_COLUMNS,
            "model_clinical_columns": model_columns, "numeric_value_columns": numeric,
            "categorical_value_columns": ["clinical_sex_recorded"],
            "categorical_values": {"clinical_sex_recorded": ["F", "M"]},
            "concept_label_columns": list(CHEXPERT_14_CONCEPTS),
            "concept_label_policy": "U-Zeros target (raw == 1) with known_/ignore_ masks, as v3",
            "concept_nullable_columns": ["nullable_" + c for c in CHEXPERT_14_CONCEPTS],
            "concept_observation_mask_columns": ["label_mask_" + c for c in CHEXPERT_14_CONCEPTS],
            "concept_state_columns": ["state_" + c for c in CHEXPERT_14_CONCEPTS],
            "outcome_column": "diagnosis_binary", "split_column": "split", "image_path_column": "image_path",
            "feature_allowlist_required": True, "metadata_is_not_model_input": metadata,
            "missing_policy": "Empty numeric cells remain missing; *_missing=1. Fit any imputation/scaling on train only.",
            "age_policy": "Approximate calendar-year age; anchor_age=91 or estimated age>89 is encoded as 91 plus topcoded=1.",
            "lab_policy": "Latest valid result in max(encounter start, t0-48h)..t0 with charttime<=storetime<=t0; "
                          "exact hadm_id or unique-encounter time linkage; 11 routine labs required.",
            "triage_policy": "MIMIC-IV-ED 2.2 triage of the latest ED stay with the same hadm_id and "
                             "encounter start <= intime <= t0; protocol ranges, else missing.",
            "lab_definitions": [{"name": lab.name, "itemid": lab.itemid, "label": lab.label, "category": lab.category,
                                 "unit": lab.unit, "accepted_units": list(lab.accepted_units),
                                 "max_value": lab.max_value, "required": lab.name in CORE} for lab in clinical.LABS],
            "triage_definitions": [{"name": name, "column": "clinical_triage_" + name, "low": float(low),
                                    "high": float(high), "unit": unit} for name, (low, high, unit) in TRIAGE.items()]}


BASE_COLUMNS = ["dicom_id", "study_id", "subject_id", "official_split", "relative_path", "hadm_id", "admission_type",
                "ViewPosition", "StudyDate", "StudyTime", "diagnosis_binary"]
EXTRA_COLUMNS = ["split", "diagnosis", "image_path", "prediction_time", "clinical_window_start", "encounter_start",
                 "radiologist_labeled", "prediction_eligible", "prediction_exclusion_reason"]


def build(args):
    started = time.time()
    output = args.output.resolve()
    output.relative_to(DATA.resolve())
    clinical.require(not output.exists(), "Output exists; use a new directory (completed builds are never overwritten)")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    inventory = verify_sources()
    frame, selection = load_selection(args.cohort.resolve())
    audit = audit_images(frame, args.workers)
    demo = demographics(frame)
    stays = pd.read_csv(ED / "ed/edstays.csv.gz", dtype=str, keep_default_na=False,
                        usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    triage_source = pd.read_csv(ED / "ed/triage.csv.gz", dtype=str, keep_default_na=False,
                                usecols=["subject_id", "stay_id", *TRIAGE])
    triage = link_triage(frame[["dicom_id", "subject_id", "hadm_id", "encounter_start", "prediction_time"]],
                         stays, triage_source)
    concepts = concept_columns(frame)

    manifest = frame[BASE_COLUMNS].copy()
    manifest = pd.concat([manifest, concepts], axis=1)
    manifest["split"] = frame["split"]
    manifest["diagnosis"] = np.where(frame["diagnosis_binary"] == "1", "deceased", "survived")
    for column in ("image_path", "prediction_time", "clinical_window_start", "encounter_start", "radiologist_labeled"):
        manifest[column] = frame[column]
    manifest["prediction_eligible"], manifest["prediction_exclusion_reason"] = "1", ""
    manifest["clinical_age_years_approx"] = demo["age_years_approx"].values
    manifest["clinical_age_is_topcoded"] = demo["age_is_topcoded"].values
    manifest["clinical_sex_recorded"] = demo["source_gender"].values
    for column in LAB_COLUMNS:
        manifest[column] = frame[column]
    for column in TRIAGE_COLUMNS:
        manifest[column] = triage[column].values
    for column in VALUE_COLUMNS:
        manifest[column + "_missing"] = (manifest[column] == "").astype(int).astype(str)

    statuses = {"clinical_triage_" + n: pd.Series(triage["status_" + n].values, index=manifest.index) for n in TRIAGE}
    sensitivity = manifest.copy()
    for lab in LAB_NAMES:
        inferred = frame["linkage_" + lab] == "inferred_unique_encounter"
        sensitivity.loc[inferred, "clinical_" + lab] = ""
        sensitivity.loc[inferred, "clinical_" + lab + "_missing"] = "1"

    lab_rows = []
    for lab in LAB_NAMES:
        for row in frame.itertuples():
            lab_rows.append({"dicom_id": row.dicom_id, "feature": "clinical_" + lab,
                             "normalized_value": getattr(row, "clinical_" + lab),
                             "source_labevent_id": getattr(row, "source_labevent_" + lab),
                             "linkage": getattr(row, "linkage_" + lab),
                             "status": "observed" if getattr(row, "clinical_" + lab) else "missing"})
    radiologist = radiologist_labels(frame)
    schema = feature_schema(concepts)
    column_order = BASE_COLUMNS + list(concepts.columns) + EXTRA_COLUMNS + VALUE_COLUMNS + MISSING_COLUMNS
    clinical.require(sorted(column_order) == sorted(manifest.columns) and len(set(column_order)) == len(column_order),
                     "Manifest column set mismatch")
    manifest, sensitivity = manifest[column_order], sensitivity[column_order]

    staging.mkdir(parents=True)
    manifest.to_csv(staging / "manifest.csv", index=False)
    sensitivity.to_csv(staging / "manifest_temporal_sensitivity.csv", index=False)
    audit.to_csv(staging / "image_audit.csv.gz", index=False)
    demo.to_csv(staging / "demographic_provenance.csv", index=False)
    pd.DataFrame(lab_rows).to_csv(staging / "lab_provenance.csv.gz", index=False)
    triage.drop(columns=TRIAGE_COLUMNS).to_csv(staging / "triage_provenance.csv.gz", index=False)
    radiologist.to_csv(staging / "radiologist_labels_test.csv", index=False)
    missingness(manifest, statuses).to_csv(staging / "missingness_summary.csv", index=False)
    clinical.write_json(staging / "feature_schema.json", schema)
    state_counts = []
    for split, part in [("all", manifest)] + list(manifest.groupby("split")):
        for name in CHEXPERT_14_CONCEPTS:
            counts = part["state_" + name].value_counts()
            state_counts.append({"split": split, "concept": name,
                                 **{s: int(counts.get(s, 0)) for s in (*STATES.values(), "not_mentioned")}})
    pd.DataFrame(state_counts).to_csv(staging / "concept_state_counts.csv", index=False)

    linked = triage["link_status"] == "linked"
    minutes = pd.to_numeric(triage.loc[linked, "minutes_intime_to_t0"])
    report = {
        "version": "v5", "protocol": "experiments/DATASET_V5.md", "protocol_sha256": clinical.sha256(PROTOCOL),
        "builder_sha256": clinical.sha256(Path(__file__)), "selector_sha256": selection["selector_sha256"],
        "selection_manifest_sha256": clinical.sha256(args.cohort.resolve() / "manifest.csv"),
        "selection_report_sha256": clinical.sha256(args.cohort.resolve() / "selection_report.json"),
        "source_inventory": inventory, "rows": len(manifest),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "deaths_by_split": {k: int(v) for k, v in manifest.groupby("split")["diagnosis_binary"].apply(lambda s: (s == "1").sum()).items()},
        "images_decoded_and_published_checksum_verified": int(audit["decoded"].sum()),
        "duplicate_image_content": int(audit["sha256"].duplicated().sum()),
        "view_counts": manifest["ViewPosition"].value_counts().to_dict(),
        "radiologist_labeled_in_test": len(radiologist),
        "triage_link_status": triage["link_status"].value_counts().to_dict(),
        "triage_value_status": {n: triage["status_" + n].value_counts().to_dict() for n in TRIAGE},
        "triage_linked_within_short_minutes": {"minutes": SHORT_TRIAGE_MINUTES,
                                               "rows": int((minutes < SHORT_TRIAGE_MINUTES).sum())},
        "observed_by_feature": {c: int((manifest[c] != "").sum()) for c in VALUE_COLUMNS},
        "sensitivity_observed_by_lab": {c: int((sensitivity[c] != "").sum()) for c in LAB_COLUMNS},
        "imputation_or_scaling_performed": False, "source_data_not_modified": True, "test_evaluated": False,
        "limitations": [
            "Population: patients with all 11 routine labs before the first AP/PA image; mortality enriched to 10%.",
            "ED-drawn labs with blank hadm_id are linked by unique-encounter time containment.",
            "Triage time is ED arrival (intime); values may be documented shortly after intime.",
            "Concept labels are report-derived CheXpert labels; radiologist labels exist only for test studies.",
            "Temperature is accepted only in the Fahrenheit range; Celsius-looking entries become missing."],
    }
    report["output_sha256"] = {p.name: clinical.sha256(p) for p in sorted(staging.iterdir())}
    clinical.write_json(staging / "report.json", report)
    staging.replace(output)
    summary = {k: report[k] for k in ("rows", "split_counts", "deaths_by_split", "images_decoded_and_published_checksum_verified",
                                      "radiologist_labeled_in_test", "triage_link_status", "triage_linked_within_short_minutes",
                                      "observed_by_feature")}
    summary["seconds"] = round(time.time() - started)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    build(parser.parse_args())
