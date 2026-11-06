#!/usr/bin/env python3
"""
Chuẩn bị tập dữ liệu kết hợp MIMIC-CXR và MIMIC-IV cho bài toán ECBM:
- Input: Ảnh X-quang lồng ngực (5.000 ca sạch, AP/PA)
- Concepts (C): 14 khái niệm CheXpert nhị phân (absent: 0, present: 1)
- Target (Y): In-hospital mortality (hospital_expire_flag: 0 - survived, 1 - deceased) từ MIMIC-IV admissions.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
ADMISSIONS_FILE = PROJECT_ROOT / "mimic-iv-3.1" / "hosp" / "admissions.csv.gz"

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

def main():
    print(f"--- Đang chuẩn bị cohort MIMIC-CXR + MIMIC-IV tại: {PROJECT_ROOT} ---")
    manifest_path = DATA_DIR / "selection_manifest.csv"
    labels_path = DATA_DIR / "selected_chexpert_labels.csv"
    meta_path = DATA_DIR / "selected_metadata.csv"

    assert manifest_path.exists(), f"Thiếu {manifest_path}"
    assert labels_path.exists(), f"Thiếu {labels_path}"
    assert admissions_file.exists() if 'admissions_file' in locals() else ADMISSIONS_FILE.exists(), f"Thiếu {ADMISSIONS_FILE}"

    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    with labels_path.open(newline="", encoding="utf-8") as f:
        label_dict = {(r["subject_id"], r["study_id"]): r for r in csv.DictReader(f)}

    with meta_path.open(newline="", encoding="utf-8") as f:
        meta_dict = {r["dicom_id"]: r for r in csv.DictReader(f)}

    # Đọc thông tin đợt nhập viện từ MIMIC-IV
    print(f"Đang đọc MIMIC-IV admissions từ {ADMISSIONS_FILE}...")
    admissions_by_subject = {}
    with gzip.open(ADMISSIONS_FILE, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            admissions_by_subject.setdefault(row["subject_id"], []).append(row)

    out_records = []
    matched_admissions = 0
    in_hospital_deaths = 0

    for idx, r in enumerate(manifest_rows):
        sub_id = r["subject_id"]
        std_id = r["study_id"]
        dic_id = r["dicom_id"]
        split = r["split"]
        # Chuẩn hóa tên split thành train / valid / test
        split_norm = "valid" if split in ("val", "validate") else split

        # Lấy ngày chụp X-quang
        meta = meta_dict.get(dic_id, {})
        study_date_str = meta.get("StudyDate", "")

        mortality = 0
        matched_hadm_id = ""
        admission_type = ""

        if sub_id in admissions_by_subject:
            matched_admissions += 1
            sub_adms = admissions_by_subject[sub_id]
            # Ưu tiên admission bao trùm ngày chụp, hoặc gần ngày chụp nhất
            try:
                study_date = datetime.strptime(study_date_str, "%Y%m%d").date()
                best_adm = min(
                    sub_adms,
                    key=lambda a: abs((datetime.strptime(a["admittime"][:10], "%Y-%m-%d").date() - study_date).days)
                )
                mortality = int(best_adm["hospital_expire_flag"])
                matched_hadm_id = best_adm["hadm_id"]
                admission_type = best_adm.get("admission_type", "")
            except Exception:
                # Nếu không parse được ngày, kiểm tra xem bệnh nhân có tử vong trong bất kỳ đợt nào
                if any(a["hospital_expire_flag"] == "1" for a in sub_adms):
                    mortality = 1

        if mortality == 1:
            in_hospital_deaths += 1

        # Xử lý 14 nhãn CheXpert (U-Zeros: 1.0 -> present (1), còn lại -> absent (0))
        chexpert_row = label_dict.get((sub_id, std_id), {})
        concept_binary_vals = []
        concept_dict = {}

        for c_name in CHEXPERT_14_CONCEPTS:
            val_raw = chexpert_row.get(c_name, "").strip()
            # U-Zeros policy
            is_present = 1 if val_raw in ("1", "1.0") else 0
            concept_binary_vals.append(is_present)
            concept_dict[c_name] = "present" if is_present == 1 else "absent"

        # Hồ sơ khái niệm (concept profile string)
        profile_str = " | ".join(f"{c}:{concept_dict[c]}" for c in CHEXPERT_14_CONCEPTS)

        record = {
            "source_index": idx,
            "subject_id": sub_id,
            "study_id": std_id,
            "dicom_id": dic_id,
            "hadm_id": matched_hadm_id,
            "admission_type": admission_type,
            "split": split_norm,
            "relative_path": r["relative_path"],
            "image_path": f"data/mimic_cxr/{r['relative_path']}",
            "StudyDate": study_date_str,
            "ViewPosition": meta.get("ViewPosition", ""),
            "diagnosis": "deceased" if mortality == 1 else "survived",
            "diagnosis_binary": mortality,
            **concept_dict,
            "concept_profile": profile_str,
        }
        out_records.append(record)

    df = pd.DataFrame(out_records)

    # Tính toán tính bất nhất của khái niệm (is_inconsistent_profile)
    # Profile có cả ca tử vong và ca sống sót
    inconsistent_mask = df.groupby("concept_profile")["diagnosis_binary"].transform("nunique") > 1
    df["is_inconsistent_profile"] = inconsistent_mask

    # Lưu manifest hoàn chỉnh
    out_manifest = DATA_DIR / "manifest_mortality.csv"
    df.to_csv(out_manifest, index=False)
    print(f"Đã lưu manifest: {out_manifest} ({len(df)} mẫu)")

    # Tạo file label_mapping.json
    label_mapping = {
        c: {"absent": 0, "present": 1} for c in CHEXPERT_14_CONCEPTS
    }
    label_mapping_file = DATA_DIR / "label_mapping.json"
    with open(label_mapping_file, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu label_mapping: {label_mapping_file}")

    # Báo cáo thống kê
    split_counts = df["split"].value_counts().to_dict()
    mortality_by_split = df.groupby(["split", "diagnosis_binary"]).size().unstack(fill_value=0).to_dict(orient="index")

    concept_positive_counts = {c: int((df[c] == "present").sum()) for c in CHEXPERT_14_CONCEPTS}

    report = {
        "dataset": "MIMIC-CXR + MIMIC-IV Cohort",
        "total_samples": len(df),
        "total_unique_patients": df["subject_id"].nunique(),
        "matched_admissions_count": matched_admissions,
        "total_mortality_positive": in_hospital_deaths,
        "mortality_rate": round(in_hospital_deaths / len(df) * 100, 2),
        "split_counts": split_counts,
        "mortality_by_split": mortality_by_split,
        "distinct_concept_profiles": int(df["concept_profile"].nunique()),
        "inconsistent_samples_count": int(df["is_inconsistent_profile"].sum()),
        "inconsistent_percentage": round(df["is_inconsistent_profile"].mean() * 100, 2),
        "concept_positive_distribution": concept_positive_counts,
    }

    report_file = DATA_DIR / "mortality_cohort_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu báo cáo thống kê: {report_file}")

    print("\n--- TỔNG KẾT ---")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Tỷ lệ tử vong chung: {in_hospital_deaths}/{len(df)} ({in_hospital_deaths/len(df)*100:.1f}%)")
    print("Phân bố theo split:")
    for sp, counts in mortality_by_split.items():
        surv = counts.get(0, 0)
        dec = counts.get(1, 0)
        print(f"  - {sp:7}: Sống={surv:4}, Tử vong={dec:3} ({dec/(surv+dec)*100:.1f}%)")
    print(f"Số profile triệu chứng khác biệt: {df['concept_profile'].nunique()} / 16384")
    print(f"Số ca rơi vào vùng bất nhất: {df['is_inconsistent_profile'].sum()} ({df['is_inconsistent_profile'].mean()*100:.1f}%)")

if __name__ == "__main__":
    main()
