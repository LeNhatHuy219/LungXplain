"""Independent validator for dataset v5 (protocol: experiments/DATASET_V5.md).

Rebuilds the complete v5 selection and every model input from the source files with
separately written code and compares it with the published dataset cell by cell:
a chunked pandas labevents scan (the selector uses pyarrow), sort/drop_duplicates
latest-result selection, an own CXR timestamp parser and age rule, a row-wise ED
triage linkage, float-based concept and triage checks, and a fresh SHA-256 + full
JPEG decode of every image. The only shared helper is the v3-validated lab value
cleaning rule (``numeric_result``). The validator writes validation_report.json;
``passed`` is true only if every check passes. Stdout is aggregate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_multimodal_clinical import LABS, numeric_result  # noqa: E402  (v3-validated value rule)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mimic_cxr"
DATASET = DATA / "multimodal_v5"
SOURCE = ROOT / "mimic-iv-3.1"
ED = ROOT / "mimic-iv-ed-2.2"
CONCEPTS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]
CORE = ["bicarbonate", "chloride", "creatinine", "glucose", "potassium", "sodium", "bun", "hematocrit",
        "hemoglobin", "platelets", "wbc"]
RANGES = {"temperature": (86.0, 113.0), "heartrate": (20.0, 300.0), "resprate": (3.0, 80.0), "o2sat": (50.0, 100.0),
          "sbp": (40.0, 300.0), "dbp": (10.0, 200.0), "acuity": (1.0, 5.0)}
QUOTA = [("test", 500, 50), ("valid", 500, 50), ("train", 4000, 400)]
FORBIDDEN = {"diagnosis_binary", "diagnosis", "subject_id", "hadm_id", "dicom_id", "study_id", "split", "prediction_time",
             "deathtime", "dischtime", "image_path", "relative_path", "radiologist_labeled", *CONCEPTS}
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
CHECKS: dict[str, dict] = {}


def check(name, condition, instances=1, **detail):
    CHECKS[name] = {"passed": bool(condition), "instances": int(instances), **detail}
    if not condition:
        print(json.dumps({"failed_check": name, **detail}, default=str), flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sums_file(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip().lstrip("*")] = parts[0]
    return out


def to_time(series):
    return pd.to_datetime(series.where(series != ""), format="ISO8601")


def cxr_time(date, clock):
    whole, _, fraction = clock.partition(".")
    if not (len(date) == 8 and date.isdigit() and whole.isdigit() and len(whole) <= 6 and (fraction == "" or fraction.isdigit())):
        raise ValueError("bad CXR time")
    whole = whole.rjust(6, "0")
    micro = (fraction + "000000")[:6]
    return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]} {whole[:2]}:{whole[2:4]}:{whole[4:]}.{micro}")


def order_key(subject, hadm, dicom):
    return hashlib.sha256(f"42|{subject}|{hadm}|{dicom}".encode()).hexdigest()


# --------------------------------------------------------------------------- selection
def rebuild_selection():
    adm = pd.read_csv(SOURCE / "hosp/admissions.csv.gz", dtype=str, keep_default_na=False)
    for c in ("admittime", "dischtime", "deathtime", "edregtime"):
        adm[c + "_t"] = to_time(adm[c])
    adm = adm[adm["admittime_t"].notna() & adm["dischtime_t"].notna()].copy()
    earlier_ed = adm["edregtime_t"].notna() & (adm["edregtime_t"] < adm["admittime_t"])
    adm["enc_start"] = adm["admittime_t"].where(~earlier_ed, adm["edregtime_t"])
    meta = pd.read_csv(DATA / "mimic-cxr-2.0.0-metadata.csv.gz", dtype=str, keep_default_na=False)
    meta = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()
    meta["t0"] = [cxr_time(d, t) for d, t in zip(meta["StudyDate"], meta["StudyTime"])]
    pairs = meta[["dicom_id", "subject_id", "study_id", "t0"]].merge(
        adm[["subject_id", "hadm_id", "enc_start", "dischtime_t"]], on="subject_id")
    pairs = pairs[(pairs["t0"] >= pairs["enc_start"]) & (pairs["t0"] <= pairs["dischtime_t"])]
    windows_per_image = pairs.groupby("dicom_id")["hadm_id"].nunique()
    first = pairs.sort_values(["hadm_id", "t0", "dicom_id"]).drop_duplicates("hadm_id", keep="first")
    first = first.merge(adm[["hadm_id", "hospital_expire_flag", "deathtime_t"]], on="hadm_id")
    ok = first["dicom_id"].map(windows_per_image).eq(1)
    ok &= (first["hospital_expire_flag"] == "1") == first["deathtime_t"].notna()
    ok &= first["deathtime_t"].isna() | (first["t0"] < first["deathtime_t"])
    chexpert = pd.read_csv(DATA / "mimic-cxr-2.0.0-chexpert.csv.gz", dtype=str, keep_default_na=False)
    ok &= pd.Series(list(zip(first["subject_id"], first["study_id"])), index=first.index).isin(
        set(zip(chexpert["subject_id"], chexpert["study_id"])))
    patients = pd.read_csv(SOURCE / "hosp/patients.csv.gz", dtype=str, keep_default_na=False).set_index("subject_id")
    demo = first[["subject_id", "t0"]].join(patients[["anchor_age", "anchor_year", "gender"]], on="subject_id")
    anchor = pd.to_numeric(demo["anchor_age"], errors="coerce")
    derived = anchor + demo["t0"].dt.year - pd.to_numeric(demo["anchor_year"], errors="coerce")
    ok &= anchor.between(0, 91) & derived.between(0, 120) & demo["gender"].isin(["F", "M"])
    unavailable = set(pd.read_csv(DATA / "unavailable_images_v5.csv", dtype=str)["relative_path"])
    ok &= ~pd.Series([f"files/p{s[:2]}/p{s}/s{st}/{d}.jpg" for s, st, d in
                      zip(first["subject_id"], first["study_id"], first["dicom_id"])], index=first.index).isin(unavailable)
    cand = first[ok].copy()
    cand["wstart"] = np.maximum(cand["enc_start"].values, (cand["t0"] - pd.Timedelta(hours=48)).values)
    pool = {"admissions_with_ap_pa_image": len(first), "eligible_first_images": len(cand)}
    return adm, cand, pool


def rebuild_labs(adm, cand, chunk_rows):
    by_item = {lab.itemid: lab for lab in LABS}
    subjects = set(cand["subject_id"])
    windows = adm[adm["subject_id"].isin(subjects)][["subject_id", "hadm_id", "enc_start", "dischtime_t"]]
    ctx = cand.set_index("hadm_id")[["subject_id", "t0", "wstart"]].rename(columns={"subject_id": "ctx_subject"})
    columns = ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"]
    kept, scanned = [], 0
    for chunk in pd.read_csv(SOURCE / "hosp/labevents.csv.gz", dtype=str, keep_default_na=False, usecols=columns,
                             chunksize=chunk_rows):
        scanned += len(chunk)
        chunk = chunk[chunk["itemid"].isin(by_item) & chunk["subject_id"].isin(subjects)]
        if chunk.empty:
            continue
        chunk = chunk.assign(ct=to_time(chunk["charttime"]), st=to_time(chunk["storetime"]))
        chunk = chunk[chunk["ct"].notna() & chunk["st"].notna() & (chunk["st"] >= chunk["ct"])]
        direct = chunk[chunk["hadm_id"] != ""].assign(target=lambda f: f["hadm_id"], link="exact_hadm_id")
        blank = chunk[chunk["hadm_id"] == ""]
        hits = blank[["labevent_id", "subject_id", "ct", "st"]].merge(windows, on="subject_id")
        hits = hits[(hits["ct"] >= hits["enc_start"]) & (hits["st"] <= hits["dischtime_t"])
                    & (hits["ct"] <= hits["dischtime_t"]) & (hits["st"] >= hits["enc_start"])]
        n = hits.groupby("labevent_id")["hadm_id"].nunique()
        single = hits[hits["labevent_id"].isin(n.index[n == 1])].drop_duplicates("labevent_id").set_index("labevent_id")["hadm_id"]
        inferred = blank[blank["labevent_id"].isin(single.index)]
        inferred = inferred.assign(target=inferred["labevent_id"].map(single), link="inferred_unique_encounter")
        events = pd.concat([direct, inferred]).join(ctx, on="target", how="inner")
        events = events[(events["subject_id"] == events["ctx_subject"]) & (events["ct"] >= events["wstart"])
                        & (events["ct"] <= events["t0"]) & (events["st"] <= events["t0"])]
        kept.append(events.drop(columns=["ctx_subject", "t0", "wstart", "charttime", "storetime"]))
    events = pd.concat(kept, ignore_index=True)
    events = events.sort_values(["target", "itemid", "ct", "st"])
    top = events.drop_duplicates(["target", "itemid"], keep="last")[["target", "itemid", "ct", "st"]]
    tied = events.merge(top, on=["target", "itemid", "ct", "st"])
    values = {}
    for (target, itemid), group in tied.groupby(["target", "itemid"], sort=False):
        lab = by_item[itemid]
        results = {numeric_result({"value": r.value, "valuenum": r.valuenum, "valueuom": r.valueuom}, lab)
                   for r in group.itertuples()}
        if len(results) == 1:
            value, quality = results.pop()
            if quality == "observed":
                first_id = min(group["labevent_id"], key=int)
                link = group.loc[group["labevent_id"] == first_id, "link"].iloc[0]
                values[(target, lab.name)] = (value, first_id, link)
    return values, events, scanned


def rebuild_choice(cand, values):
    for lab in LABS:
        cand["lab_" + lab.name] = [values.get((h, lab.name), ("",))[0] for h in cand["hadm_id"]]
    full = cand[np.logical_and.reduce([cand["lab_" + l] != "" for l in CORE])].copy()
    labeled = set(pd.read_csv(DATA / "mimic-cxr-2.1.0-test-set-labeled.csv", dtype=str)["study_id"])
    full["rad"] = full["study_id"].isin(labeled)
    full["key"] = [order_key(s, h, d) for s, h, d in zip(full["subject_id"], full["hadm_id"], full["dicom_id"])]
    per_patient = full.sort_values(["subject_id", "rad", "key"], ascending=[True, False, True]).drop_duplicates("subject_id")
    per_patient = per_patient.sort_values("key")
    stats = {"admissions_with_all_11_core_labs": len(full), "eligible_patients": len(per_patient),
             "eligible_patient_deaths": int((per_patient["hospital_expire_flag"] == "1").sum()),
             "eligible_patients_radiologist_labeled": int(per_patient["rad"].sum())}
    assignment, taken = {}, set()
    for split, rows, deaths in QUOTA:
        remaining = per_patient[~per_patient["subject_id"].isin(taken)]
        forced = remaining[remaining["rad"]] if split == "test" else remaining.iloc[:0]
        picks = list(forced["subject_id"])
        for outcome, quota in (("1", deaths), ("0", rows - deaths)):
            need = quota - int((forced["hospital_expire_flag"] == outcome).sum())
            if need < 0:
                raise ValueError("forced studies exceed quota")
            pool = remaining[(remaining["hospital_expire_flag"] == outcome) & ~remaining["rad"]] if split == "test" \
                else remaining[remaining["hospital_expire_flag"] == outcome]
            picks += list(pool["subject_id"].iloc[:need])
        for subject in picks:
            assignment[subject] = split
        taken |= set(picks)
    chosen = per_patient[per_patient["subject_id"].isin(assignment)].copy()
    chosen["split"] = chosen["subject_id"].map(assignment)
    return chosen, stats


# --------------------------------------------------------------------------- triage / concepts
def float_or_none(text):
    text = text.strip()
    return float(text) if NUMBER.fullmatch(text) else None


def rebuild_triage(manifest):
    stays = pd.read_csv(ED / "ed/edstays.csv.gz", dtype=str, keep_default_na=False,
                        usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    triage = pd.read_csv(ED / "ed/triage.csv.gz", dtype=str, keep_default_na=False,
                         usecols=["subject_id", "stay_id", *RANGES])
    by_hadm = defaultdict(list)
    for s in stays.itertuples():
        if s.hadm_id:
            by_hadm[s.hadm_id].append(s)
    tri = {t.stay_id: t for t in triage.itertuples()}
    out = []
    for row in manifest.itertuples():
        start, t0 = pd.Timestamp(row.encounter_start), pd.Timestamp(row.prediction_time)
        linked = by_hadm.get(row.hadm_id, [])
        if any(s.subject_id != row.subject_id for s in linked):
            raise ValueError("ED stay subject mismatch")
        inside = [s for s in linked if s.intime and start <= pd.Timestamp(s.intime) <= t0]
        result = {"dicom_id": row.dicom_id, "stay": "", "status": ""}
        values = {name: None for name in RANGES}
        if not linked:
            result["status"] = "no_linked_ed_stay"
        elif not inside:
            result["status"] = "ed_stay_outside_window"
        else:
            latest = max(pd.Timestamp(s.intime) for s in inside)
            best = [s for s in inside if pd.Timestamp(s.intime) == latest]
            if len(best) > 1:
                result["status"] = "ambiguous_latest_ed_stay"
            elif best[0].stay_id not in tri:
                result["status"], result["stay"] = "no_triage_row", best[0].stay_id
            else:
                result["status"], result["stay"] = "linked", best[0].stay_id
                source = tri[best[0].stay_id]
                for name, (low, high) in RANGES.items():
                    number = float_or_none(getattr(source, name))
                    if number is not None and low <= number <= high and (name != "acuity" or number.is_integer()):
                        values[name] = number
                if values["sbp"] is not None and values["dbp"] is not None and values["dbp"] >= values["sbp"]:
                    values["sbp"] = values["dbp"] = None
        result.update(values)
        out.append(result)
    return pd.DataFrame(out).set_index("dicom_id")


# --------------------------------------------------------------------------- main
def main(args):
    started = time.time()
    directory = args.dataset.resolve()
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    listed = report["output_sha256"]
    present = {p.name for p in directory.iterdir()} - {"report.json", "validation_report.json"}
    check("output_files_exactly_listed", present == set(listed), len(listed))
    check("output_hashes_match", all(digest(directory / name) == h for name, h in listed.items()), len(listed))
    check("builder_and_protocol_recorded", all(report.get(k) for k in ("builder_sha256", "protocol_sha256", "selector_sha256")))
    check("protocol_unchanged_since_build", digest(ROOT / "experiments/DATASET_V5.md") == report["protocol_sha256"])

    mimic_sums, ed_sums = sums_file(SOURCE / "SHA256SUMS.txt"), sums_file(ED / "SHA256SUMS.txt")
    cxr_sums = sums_file(DATA / "mimic-cxr-jpg-2.1.0-SHA256SUMS.txt")
    sources = [(SOURCE, mimic_sums, ["hosp/admissions.csv.gz", "hosp/patients.csv.gz", "hosp/labevents.csv.gz"]),
               (ED, ed_sums, ["ed/edstays.csv.gz", "ed/triage.csv.gz"]),
               (DATA, cxr_sums, ["mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-chexpert.csv.gz",
                                 "mimic-cxr-2.0.0-split.csv.gz", "mimic-cxr-2.1.0-test-set-labeled.csv"])]
    source_ok = [digest(root / name) == sums.get(name) for root, sums, names in sources for name in names]
    check("source_files_match_distributed_checksums", all(source_ok), len(source_ok))

    manifest = pd.read_csv(directory / "manifest.csv", dtype=str, keep_default_na=False)
    sensitivity = pd.read_csv(directory / "manifest_temporal_sensitivity.csv", dtype=str, keep_default_na=False)
    schema = json.loads((directory / "feature_schema.json").read_text())
    n = len(manifest)
    check("exactly_5000_rows", n == 5000, n)
    check("unique_patients_admissions_images_studies",
          all(manifest[k].nunique() == n for k in ("subject_id", "hadm_id", "dicom_id", "study_id")), 4 * n)
    check("split_sizes", manifest["split"].value_counts().to_dict() == {"train": 4000, "valid": 500, "test": 500})
    deaths = {s: int((p["diagnosis_binary"] == "1").sum()) for s, p in manifest.groupby("split")}
    check("deaths_by_split", deaths == {"train": 400, "valid": 50, "test": 50}, detail=deaths)
    check("diagnosis_text_matches_outcome", ((manifest["diagnosis"] == "deceased") == (manifest["diagnosis_binary"] == "1")).all(), n)
    check("relative_path_matches_ids", all(r.relative_path == f"files/p{r.subject_id[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg"
                                           for r in manifest.itertuples()), n)

    # images
    def image_ok(row):
        path = ROOT / row.image_path
        if not path.is_file() or not path.resolve().is_relative_to(DATA.resolve()):
            return False
        if digest(path) != cxr_sums.get(row.relative_path):
            return False
        try:
            with Image.open(path) as image:
                image.load()
                return image.format == "JPEG" and min(image.size) > 0
        except OSError:
            return False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        image_results = list(pool.map(image_ok, manifest.itertuples()))
    valid_images = int(sum(image_results))
    check("images_checksum_and_full_decode", valid_images == n, n, valid_images=valid_images)

    # selection rebuilt from source
    adm, cand, pool_stats = rebuild_selection()
    values, events, scanned = rebuild_labs(adm, cand, args.chunk_rows)
    chosen, stats = rebuild_choice(cand, values)
    selection_report = json.loads((args.cohort.resolve() / "selection_report.json").read_text())
    selection = selection_report["flow"]
    check("unavailable_image_list_bound_to_selection",
          selection_report.get("unavailable_images_sha256") == digest(DATA / "unavailable_images_v5.csv"))
    for key, value in stats.items():
        check("pool_" + key, selection.get(key) == value, 1, rebuilt=value, selector=selection.get(key))
    published = set(zip(manifest["subject_id"], manifest["hadm_id"], manifest["dicom_id"], manifest["split"]))
    rebuilt = set(zip(chosen["subject_id"], chosen["hadm_id"], chosen["dicom_id"], chosen["split"]))
    check("selected_set_and_splits_reproduced", published == rebuilt, n,
          published_only=len(published - rebuilt), rebuilt_only=len(rebuilt - published))
    c = chosen.set_index("dicom_id").reindex(manifest["dicom_id"])
    check("outcome_matches_admission_flag", (c["hospital_expire_flag"].values == manifest["diagnosis_binary"].values).all(), n)
    check("prediction_time_reproduced", all(pd.Timestamp(a) == b for a, b in zip(manifest["prediction_time"], c["t0"])), n)
    check("encounter_start_reproduced", all(pd.Timestamp(a) == b for a, b in zip(manifest["encounter_start"], c["enc_start"])), n)
    check("radiologist_flag_reproduced", ((manifest["radiologist_labeled"] == "1").values == c["rad"].values).all(), n)

    lab_cells = mismatched = 0
    provenance = pd.read_csv(directory / "lab_provenance.csv.gz", dtype=str, keep_default_na=False)
    prov = provenance.set_index(["dicom_id", "feature"])
    for lab in LABS:
        for row in manifest.itertuples():
            lab_cells += 1
            expected = values.get((row.hadm_id, lab.name), ("", "", ""))
            got = getattr(row, "clinical_" + lab.name)
            p = prov.loc[(row.dicom_id, "clinical_" + lab.name)]
            if (got, p["source_labevent_id"], p["linkage"]) != (expected[0], expected[1], expected[2]):
                mismatched += 1
    check("lab_values_ids_linkage_reproduced", mismatched == 0, lab_cells, mismatched=mismatched)
    check("required_labs_complete", all((manifest["clinical_" + l] != "").all() for l in CORE), 11 * n)
    used = set(provenance.loc[provenance["source_labevent_id"] != "", "source_labevent_id"])
    ev = events.drop_duplicates("labevent_id").set_index("labevent_id")
    ctx = manifest.set_index("hadm_id")
    temporal_ok = True
    for row in provenance[provenance["source_labevent_id"] != ""].itertuples():
        e = ev.loc[row.source_labevent_id]
        m = ctx.loc[e["target"]]
        temporal_ok &= (pd.Timestamp(m["clinical_window_start"]) <= e["ct"] <= e["st"] <= pd.Timestamp(m["prediction_time"]))
    check("every_used_lab_event_inside_window_and_stored_before_t0", temporal_ok, len(used))

    # sensitivity manifest
    sens_ok = list(sensitivity.columns) == list(manifest.columns) and (sensitivity["dicom_id"] == manifest["dicom_id"]).all()
    lab_cols = {"clinical_" + lab.name for lab in LABS} | {"clinical_" + lab.name + "_missing" for lab in LABS}
    for column in manifest.columns:
        if column not in lab_cols:
            sens_ok &= (sensitivity[column] == manifest[column]).all()
    for lab in LABS:
        inferred = prov.xs("clinical_" + lab.name, level="feature").reindex(manifest["dicom_id"])["linkage"].eq("inferred_unique_encounter").values
        expect = manifest["clinical_" + lab.name].where(~inferred, "")
        sens_ok &= (sensitivity["clinical_" + lab.name] == expect).all()
        sens_ok &= (sensitivity["clinical_" + lab.name + "_missing"] == (expect == "").astype(int).astype(str)).all()
    check("sensitivity_manifest_is_exact_hadm_only", sens_ok, n)

    # triage
    triage = rebuild_triage(manifest).reindex(manifest["dicom_id"])
    tprov = pd.read_csv(directory / "triage_provenance.csv.gz", dtype=str, keep_default_na=False).set_index("dicom_id").reindex(manifest["dicom_id"])
    check("triage_link_status_reproduced", (tprov["link_status"].values == triage["status"].values).all(), n)
    check("triage_stay_reproduced", (tprov["ed_stay_id"].values == triage["stay"].values).all(), n)
    triage_bad = 0
    for name in RANGES:
        got = pd.to_numeric(manifest["clinical_triage_" + name].replace("", np.nan)).values
        exp = triage[name].astype(float).values
        triage_bad += int(np.sum(~((np.isnan(got) & np.isnan(exp)) | np.isclose(got, exp, rtol=0, atol=1e-9))))
    check("triage_values_reproduced", triage_bad == 0, n * len(RANGES), mismatched=triage_bad)
    linked = tprov["link_status"] == "linked"
    intime = pd.to_datetime(tprov.loc[linked, "ed_intime"], format="ISO8601")
    check("triage_intime_not_after_t0", (intime.values <= pd.to_datetime(manifest.loc[linked.values, "prediction_time"], format="ISO8601").values).all(), int(linked.sum()))

    # concepts
    chexpert = pd.read_csv(DATA / "mimic-cxr-2.0.0-chexpert.csv.gz", dtype=str, keep_default_na=False).set_index(["subject_id", "study_id"])
    src = chexpert.loc[list(zip(manifest["subject_id"], manifest["study_id"]))]
    concept_ok = True
    for name in CONCEPTS:
        raw = [float(v) if v != "" else None for v in src[name]]
        concept_ok &= list(manifest["raw_" + name]) == list(src[name])
        concept_ok &= list(manifest[name]) == ["1" if v == 1.0 else "0" for v in raw]
        concept_ok &= list(manifest["ignore_" + name]) == ["1" if v == -1.0 else "0" for v in raw]
        concept_ok &= list(manifest["known_" + name]) == ["1" if v in (0.0, 1.0) else "0" for v in raw]
        concept_ok &= list(manifest["label_mask_" + name]) == list(manifest["known_" + name])
        concept_ok &= list(manifest["nullable_" + name]) == [("1" if v == 1.0 else "0") if v in (0.0, 1.0) else "" for v in raw]
        concept_ok &= list(manifest["state_" + name]) == [{1.0: "positive", 0.0: "explicit_negative", -1.0: "uncertain"}.get(v, "not_mentioned") for v in raw]
    check("concept_columns_reproduced_from_chexpert", concept_ok, 7 * 14 * n)
    radiologist = pd.read_csv(directory / "radiologist_labels_test.csv", dtype=str, keep_default_na=False)
    rsrc = pd.read_csv(DATA / "mimic-cxr-2.1.0-test-set-labeled.csv", dtype=str, keep_default_na=False).set_index("study_id")
    rad_ok = set(radiologist["study_id"]) == set(manifest.loc[manifest["radiologist_labeled"] == "1", "study_id"]) \
        and (radiologist["split"] == "test").all()
    for _, r in radiologist.iterrows():
        source_row = rsrc.loc[r["study_id"]]
        rad_ok &= all(r[c] == source_row["Airspace Opacity" if c == "Lung Opacity" else c] for c in CONCEPTS)
    check("radiologist_labels_match_source_and_test_only", rad_ok, len(radiologist))

    # demographics
    patients = pd.read_csv(SOURCE / "hosp/patients.csv.gz", dtype=str, keep_default_na=False).set_index("subject_id")
    demo_ok = True
    for row in manifest.itertuples():
        p = patients.loc[row.subject_id]
        age = int(p["anchor_age"]) + pd.Timestamp(row.prediction_time).year - int(p["anchor_year"])
        top = int(p["anchor_age"]) == 91 or age > 89
        demo_ok &= (row.clinical_age_years_approx == ("91" if top else str(age))
                    and row.clinical_age_is_topcoded == str(int(top)) and row.clinical_sex_recorded == p["gender"])
    check("demographics_reproduced", demo_ok, n)

    # schema / leakage
    model = schema["model_clinical_columns"]
    check("model_columns_exist_and_prefixed", all(c in manifest.columns and c.startswith("clinical_") for c in model), len(model))
    check("no_forbidden_or_label_predictor", not (set(model) & FORBIDDEN) and not any(
        c.startswith(("raw_", "known_", "ignore_", "nullable_", "label_mask_", "state_")) for c in model), len(model))
    flags_ok = all((manifest[c + "_missing"] == (manifest[c] == "").astype(int).astype(str)).all() for c in schema["clinical_value_columns"])
    check("missing_flags_consistent", flags_ok, n * len(schema["clinical_value_columns"]))
    numeric_ok = all(pd.to_numeric(manifest[c].replace("", np.nan), errors="coerce").notna().sum() == (manifest[c] != "").sum()
                     for c in schema["numeric_value_columns"])
    check("numeric_columns_parse", numeric_ok and set(manifest["clinical_sex_recorded"]) <= {"F", "M"}, len(schema["numeric_value_columns"]))
    check("concept_schema_14", schema["concept_label_columns"] == CONCEPTS)

    passed = all(v["passed"] for v in CHECKS.values())
    result = {"passed": passed, "dataset_version": "v5", "active_rows": n, "excluded_rows": 0,
              "checks": CHECKS, "check_instances": int(sum(v["instances"] for v in CHECKS.values())),
              "exact_5000_validation": {"valid_images": valid_images, "required": 5000},
              "images_checksum_rechecked": valid_images, "full_source_scan_repeated": True,
              "labevents_rows_scanned": scanned, "rebuilt_pool": {**pool_stats, **stats},
              "independent_selection_method": "pandas chunked labevents scan, sort/drop_duplicates latest selection, "
                                              "row-wise ED triage linkage, float-based concept checks",
              "report_sha256": digest(report_path), "validator_sha256": digest(Path(__file__)),
              "primary_manifest_sha256": digest(directory / "manifest.csv"),
              "sensitivity_manifest_sha256": digest(directory / "manifest_temporal_sensitivity.csv"),
              "test_evaluated": False, "seconds": round(time.time() - started)}
    if args.write:
        tmp = directory / "validation_report.json.partial"
        tmp.write_text(json.dumps(result, indent=2, default=str) + "\n")
        tmp.replace(directory / "validation_report.json")
    print(json.dumps({k: result[k] for k in ("passed", "active_rows", "check_instances", "exact_5000_validation",
                                             "rebuilt_pool", "seconds")}, indent=1))
    print(json.dumps({"failed_checks": [k for k, v in CHECKS.items() if not v["passed"]], "checks": len(CHECKS)}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cohort", type=Path, default=DATA / "cohort_v5r2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--write", action="store_true", help="Write validation_report.json into the dataset directory")
    raise SystemExit(main(parser.parse_args()))
