"""Dataset v5: reselect exactly 5000 patients for a more complete multimodal cohort.

Protocol (fixed before selection): experiments/DATASET_V5.md. Changes from v3:
ED labs with an empty source hadm_id are linked when the target encounter is the
only encounter of the patient containing both charttime and storetime; the lab
window is max(encounter start, t0 - 48 h) .. t0; all 11 routine labs are required;
studies with MIMIC-CXR radiologist labels go to the test split; ED triage vitals
are added in the build stage. Image/encounter/outcome rules and value cleaning are
the v3 rules (helpers imported from build_multimodal_clinical, not modified).

Stage ``select`` needs no network and writes data/mimic_cxr/cohort_v5/. Images
that are not already on disk are listed in download_list.txt for the user's own
PhysioNet download. Patient-level files stay under the ignored data/ directory;
stdout is aggregate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multimodal_clinical as clinical  # noqa: E402
from download_images import load_checksums  # noqa: E402

ROOT = clinical.ROOT
DATA = ROOT / "data/mimic_cxr"
SOURCE = ROOT / "mimic-iv-3.1"
COHORT = DATA / "cohort_v5r2"  # history: cohort_v5 (latest-time bug), cohort_v5r (one image HTTP 404)
UNAVAILABLE = DATA / "unavailable_images_v5.csv"  # images the official distribution does not serve
SEED = 42
LOOKBACK = timedelta(hours=48)
CORE = ("bicarbonate", "chloride", "creatinine", "glucose", "potassium", "sodium", "bun",
        "hematocrit", "hemoglobin", "platelets", "wbc")
LAB_NAMES = tuple(lab.name for lab in clinical.LABS)
TARGET = {"test": (500, 50), "valid": (500, 50), "train": (4000, 400)}
LOCAL_IMAGE_ROOTS = (DATA / "dataset", DATA / "cohort_5000", DATA / "cohort_v5", DATA / "cohort_v5r")  # checksum-verified reuse only
PROTOCOL = ROOT / "experiments/DATASET_V5.md"


def stable_key(subject, hadm, dicom):
    return hashlib.sha256(f"{SEED}|{subject}|{hadm}|{dicom}".encode()).hexdigest()


def relative_path(subject, study, dicom):
    return f"files/p{subject[:2]}/p{subject}/s{study}/{dicom}.jpg"


def load_encounters():
    adm = pd.read_csv(SOURCE / "hosp/admissions.csv.gz", dtype=str, keep_default_na=False)
    for column in ("admittime", "dischtime", "deathtime", "edregtime"):
        adm[column + "_ts"] = pd.to_datetime(adm[column].replace("", None), format="ISO8601")
    adm = adm[adm["admittime_ts"].notna() & adm["dischtime_ts"].notna()].copy()
    adm["start"] = adm[["admittime_ts", "edregtime_ts"]].min(axis=1)
    adm["end"] = adm["dischtime_ts"]
    return adm


def first_images(adm, flow):
    meta = pd.read_csv(DATA / "mimic-cxr-2.0.0-metadata.csv.gz", dtype=str, keep_default_na=False)
    meta = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()
    flow["ap_pa_images"] = len(meta)
    meta["t0"] = [pd.Timestamp(clinical.study_timestamp(d, t)) for d, t in zip(meta["StudyDate"], meta["StudyTime"])]
    joined = meta.merge(adm[["subject_id", "hadm_id", "start", "end"]], on="subject_id")
    joined = joined[(joined["t0"] >= joined["start"]) & (joined["t0"] <= joined["end"])]
    first = joined.sort_values(["hadm_id", "t0", "dicom_id"]).groupby("hadm_id", sort=False).head(1)
    flow["admissions_with_ap_pa_image"] = len(first)
    return first.drop(columns=["start", "end"])


def eligibility(first, adm, flow):
    a = adm.set_index("hadm_id")
    c = first.join(a[["admission_type", "hospital_expire_flag", "deathtime_ts", "start", "end"]], on="hadm_id")
    status = pd.Series("eligible", index=c.index)
    died = c["hospital_expire_flag"] == "1"
    status[died != c["deathtime_ts"].notna()] = "outcome_death_time_mismatch"
    status[(status == "eligible") & c["deathtime_ts"].notna() & (c["t0"] >= c["deathtime_ts"])] = "not_before_recorded_death"
    windows = adm[["subject_id", "hadm_id", "start", "end"]]
    hits = c[["dicom_id", "subject_id", "t0"]].merge(windows, on="subject_id")
    hits = hits[(hits["t0"] >= hits["start"]) & (hits["t0"] <= hits["end"])].groupby("dicom_id")["hadm_id"].nunique()
    status[(status == "eligible") & (c["dicom_id"].map(hits) != 1).values] = "overlapping_or_wrong_encounter"
    chexpert = pd.read_csv(DATA / "mimic-cxr-2.0.0-chexpert.csv.gz", dtype=str, keep_default_na=False)
    labelled = set(zip(chexpert["subject_id"], chexpert["study_id"]))
    no_label = ~pd.Series([k in labelled for k in zip(c["subject_id"], c["study_id"])], index=c.index)
    status[(status == "eligible") & no_label] = "missing_concept_source_row"
    clinical.require(UNAVAILABLE.is_file(), "Missing the fixed list of unavailable images")
    unavailable = set(pd.read_csv(UNAVAILABLE, dtype=str)["relative_path"])
    paths = pd.Series([relative_path(s, st, d) for s, st, d in zip(c["subject_id"], c["study_id"], c["dicom_id"])], index=c.index)
    status[(status == "eligible") & paths.isin(unavailable)] = "image_unavailable_in_distribution"
    patients = pd.read_csv(SOURCE / "hosp/patients.csv.gz", dtype=str, keep_default_na=False).set_index("subject_id")
    bad_demo = []
    for index, row in c[status == "eligible"].iterrows():
        try:
            patient = patients.loc[row["subject_id"]].to_dict()
            clinical.age_features(patient, row["t0"].to_pydatetime())
            clinical.require(patient["gender"] in {"F", "M"}, "Unknown sex")
        except (KeyError, ValueError):
            bad_demo.append(index)
    status[bad_demo] = "invalid_demographics"
    flow["first_image_status"] = status.value_counts().to_dict()
    c = c[status == "eligible"].copy()
    c["window_start"] = np.maximum(c["start"].values, (c["t0"] - LOOKBACK).values)
    return c


def latest_by_time(events):
    """Events tied at the latest (charttime, storetime) pair per (target, itemid).

    The pair is ordered lexicographically. The first selection (cohort_v5) took the
    maximum of each column separately, which drops a group whenever an earlier draw
    was stored later than the latest draw; that rule is not the protocol and was
    corrected on 15/11/2026 before any build or model use (see DATASET_V5.md)."""
    keys = ["target", "itemid", "chart", "store"]
    ordered = events.sort_values(keys)
    tail = ordered.groupby(["target", "itemid"], sort=False).tail(1)[keys]
    return ordered.merge(tail, on=keys)


def lab_features(candidates, adm, flow):
    """One labevents pass; v3 value rules; v5 linkage and 48 h window."""
    context = candidates.set_index("hadm_id")[["subject_id", "t0", "window_start"]]
    subjects = pa.array(sorted(set(candidates["subject_id"].astype(np.int64))), type=pa.int64())
    items = pa.array(sorted(int(x) for x in clinical.BY_ITEM), type=pa.int64())
    windows = adm[adm["subject_id"].isin(set(candidates["subject_id"]))][["subject_id", "hadm_id", "start", "end"]]
    columns = ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"]
    types = {c: pa.string() for c in columns}
    reader = pacsv.open_csv(SOURCE / "hosp/labevents.csv.gz", read_options=pacsv.ReadOptions(block_size=64 << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=columns, column_types=types,
                                                                 strings_can_be_null=False))
    kept, scanned, started = [], 0, time.time()
    for batch in reader:
        scanned += batch.num_rows
        subject = pc.cast(batch["subject_id"], pa.int64())
        item = pc.cast(batch["itemid"], pa.int64())
        mask = pc.and_(pc.is_in(subject, subjects), pc.is_in(item, items))
        if not pc.any(mask).as_py():
            continue
        e = pa.Table.from_batches([batch]).filter(mask).to_pandas()
        e["chart"] = pd.to_datetime(e["charttime"].replace("", None), format="ISO8601")
        e["store"] = pd.to_datetime(e["storetime"].replace("", None), format="ISO8601")
        e = e[e["chart"].notna() & e["store"].notna() & (e["store"] >= e["chart"])]
        exact = e[e["hadm_id"] != ""].copy()
        exact["target"], exact["linkage"] = exact["hadm_id"], "exact_hadm_id"
        blank = e[e["hadm_id"] == ""]
        if len(blank):
            w = blank[["labevent_id", "subject_id", "chart", "store"]].merge(windows, on="subject_id")
            inside = w[(w["chart"] >= w["start"]) & (w["chart"] <= w["end"]) & (w["store"] >= w["start"]) & (w["store"] <= w["end"])]
            unique = inside.groupby("labevent_id")["hadm_id"].agg(["nunique", "first"])
            unique = unique[unique["nunique"] == 1]["first"]
            inferred = blank[blank["labevent_id"].isin(unique.index)].copy()
            inferred["target"], inferred["linkage"] = inferred["labevent_id"].map(unique), "inferred_unique_encounter"
            e = pd.concat([exact, inferred])
        else:
            e = exact
        e = e[e["target"].isin(context.index)]
        if e.empty:
            continue
        ctx = context.loc[e["target"]].reset_index(drop=True)
        e = e.reset_index(drop=True)
        ok = (ctx["subject_id"].values == e["subject_id"].values) & (e["chart"].values >= ctx["window_start"].values) \
            & (e["chart"].values <= ctx["t0"].values) & (e["store"].values <= ctx["t0"].values)
        kept.append(e[ok][["labevent_id", "target", "itemid", "chart", "store", "value", "valuenum", "valueuom", "linkage"]])
        if scanned // 20_000_000 != (scanned - batch.num_rows) // 20_000_000:
            print(json.dumps({"labevents_rows_scanned": scanned, "seconds": round(time.time() - started)}), flush=True)
    events = pd.concat(kept, ignore_index=True)
    clinical.require(not events.duplicated(["labevent_id", "target"]).any(), "Duplicate source lab event")
    flow["labevents_rows_scanned"] = scanned
    flow["eligible_timed_events"] = len(events)
    flow["eligible_timed_events_by_linkage"] = events["linkage"].value_counts().to_dict()
    values = {}
    latest = latest_by_time(events)
    status_counts = Counter()
    for (hadm, itemid), group in latest.groupby(["target", "itemid"]):
        lab = clinical.BY_ITEM[itemid]
        signatures = {clinical.numeric_result({"value": r.value, "valuenum": r.valuenum, "valueuom": r.valueuom}, lab)
                      for r in group.itertuples()}
        if len(signatures) > 1:
            status_counts["conflicting_latest_events"] += 1
            continue
        value, quality = signatures.pop()
        status_counts[quality] += 1
        if quality == "observed":
            chosen = group.sort_values("labevent_id", key=lambda s: s.astype(np.int64)).iloc[0]
            values[(hadm, lab.name)] = (value, chosen["labevent_id"], chosen["linkage"], len(group))
    flow["latest_event_status"] = dict(status_counts)
    return values


def choose(candidates, values, flow):
    for lab in LAB_NAMES:
        candidates["clinical_" + lab] = [values.get((h, lab), ("",))[0] for h in candidates["hadm_id"]]
    complete = np.all([candidates["clinical_" + lab] != "" for lab in CORE], axis=0)
    flow["admissions_with_all_11_core_labs"] = int(complete.sum())
    eligible = candidates[complete].copy()
    labeled = set(pd.read_csv(DATA / "mimic-cxr-2.1.0-test-set-labeled.csv", dtype=str)["study_id"])
    eligible["radiologist_labeled"] = eligible["study_id"].isin(labeled).astype(int)
    eligible["order"] = [stable_key(s, h, d) for s, h, d in zip(eligible["subject_id"], eligible["hadm_id"], eligible["dicom_id"])]
    eligible = eligible.sort_values(["subject_id", "radiologist_labeled", "order"], ascending=[True, False, True])
    patients = eligible.groupby("subject_id", sort=False).head(1).copy()
    patients["diagnosis_binary"] = patients["hospital_expire_flag"]
    flow["eligible_patients"] = len(patients)
    flow["eligible_patient_deaths"] = int((patients["diagnosis_binary"] == "1").sum())
    flow["eligible_patients_radiologist_labeled"] = int(patients["radiologist_labeled"].sum())
    patients = patients.sort_values("order")
    chosen, used = [], set()
    for split, (rows, deaths) in TARGET.items():
        pool = patients[~patients["subject_id"].isin(used)]
        forced = pool[pool["radiologist_labeled"] == 1] if split == "test" else pool.iloc[:0]
        need = {"1": deaths - int((forced["diagnosis_binary"] == "1").sum()),
                "0": rows - deaths - int((forced["diagnosis_binary"] == "0").sum())}
        clinical.require(min(need.values()) >= 0, "Radiologist-labeled studies exceed a test stratum")
        rest = pool[~pool["subject_id"].isin(set(forced["subject_id"]))]
        parts = [forced] + [rest[rest["diagnosis_binary"] == y].head(n) for y, n in need.items()]
        block = pd.concat(parts).assign(split=split)
        clinical.require(len(block) == rows and int((block["diagnosis_binary"] == "1").sum()) == deaths,
                         f"Not enough eligible patients for split {split}")
        used |= set(block["subject_id"])
        chosen.append(block)
    return pd.concat(chosen).sort_values(["split", "order"]).reset_index(drop=True)


def select(args):
    output = args.output.resolve()
    output.relative_to(DATA)
    clinical.require(not output.exists() or not any(output.iterdir()), "Use a new empty output directory")
    started, flow = time.time(), {}
    adm = load_encounters()
    candidates = eligibility(first_images(adm, flow), adm, flow)
    values = lab_features(candidates, adm, flow)
    cohort = choose(candidates, values, flow)
    split_src = pd.read_csv(DATA / "mimic-cxr-2.0.0-split.csv.gz", dtype=str).set_index("dicom_id")["split"]
    cohort["official_split"] = cohort["dicom_id"].map(split_src)
    cohort["relative_path"] = [relative_path(s, st, d) for s, st, d in zip(cohort["subject_id"], cohort["study_id"], cohort["dicom_id"])]
    sums = load_checksums()
    image_paths, download = [], []
    for name in cohort["relative_path"]:
        clinical.require(name in sums, "Selected image lacks a published checksum")
        local = next((root / name for root in LOCAL_IMAGE_ROOTS if (root / name).is_file()
                      and clinical.sha256(root / name) == sums[name]), None)
        if local is None:
            local = output / name
            download.append(name)
        image_paths.append(local.relative_to(ROOT).as_posix())
    cohort["image_path"] = image_paths
    cohort["prediction_time"] = [clinical.iso(t.to_pydatetime()) for t in cohort["t0"]]
    cohort["encounter_start"] = [clinical.iso(t.to_pydatetime()) for t in cohort["start"]]
    cohort["clinical_window_start"] = [clinical.iso(pd.Timestamp(t).to_pydatetime()) for t in cohort["window_start"]]
    for lab in LAB_NAMES:
        record = [values.get((h, lab)) for h in cohort["hadm_id"]]
        cohort["source_labevent_" + lab] = [r[1] if r else "" for r in record]
        cohort["linkage_" + lab] = [r[2] if r else "" for r in record]
    keep = ["dicom_id", "study_id", "subject_id", "hadm_id", "official_split", "relative_path", "image_path",
            "admission_type", "ViewPosition", "StudyDate", "StudyTime", "diagnosis_binary", "split",
            "radiologist_labeled", "prediction_time", "encounter_start", "clinical_window_start"]
    keep += ["clinical_" + lab for lab in LAB_NAMES] + ["source_labevent_" + l for l in LAB_NAMES] + ["linkage_" + l for l in LAB_NAMES]
    output.mkdir(parents=True, exist_ok=True)
    cohort[keep].to_csv(output / "manifest.csv", index=False)
    (output / "download_list.txt").write_text("".join(name + "\n" for name in download))
    linkage = Counter(v for l in LAB_NAMES for v in cohort["linkage_" + l] if v)
    report = {"protocol": "experiments/DATASET_V5.md", "protocol_sha256": clinical.sha256(PROTOCOL),
              "selector_sha256": clinical.sha256(Path(__file__)), "unavailable_images_sha256": clinical.sha256(UNAVAILABLE), "seed": SEED, "lookback_hours": 48,
              "required_labs": list(CORE), "flow": flow,
              "selected_rows": len(cohort), "selected_patients": int(cohort["subject_id"].nunique()),
              "split_counts": cohort["split"].value_counts().to_dict(),
              "deaths_by_split": cohort.groupby("split")["diagnosis_binary"].apply(lambda s: int((s == "1").sum())).to_dict(),
              "radiologist_labeled_selected": int(cohort["radiologist_labeled"].sum()),
              "radiologist_labeled_in_test": int(cohort.loc[cohort["split"] == "test", "radiologist_labeled"].sum()),
              "view_counts": cohort["ViewPosition"].value_counts().to_dict(),
              "albumin_observed": int((cohort["clinical_albumin"] != "").sum()),
              "selected_lab_values_by_linkage": dict(linkage),
              "images_already_on_disk": len(cohort) - len(download), "images_to_download": len(download),
              "v3_patients_reselected": int(cohort["subject_id"].isin(set(pd.read_csv(
                  DATA / "multimodal_v3/manifest.csv", dtype=str, usecols=["subject_id"])["subject_id"])).sum()),
              "test_evaluated": False, "seconds": round(time.time() - started)}
    clinical.write_json(output / "selection_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "flow"}, indent=1))
    print(json.dumps({"flow": flow}, indent=1, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("select")
    s.add_argument("--output", type=Path, default=COHORT)
    args = parser.parse_args()
    {"select": select}[args.stage](args)
