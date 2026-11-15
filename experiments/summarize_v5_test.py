"""Aggregate the single, locked v5 test evaluation for the paper (no new model fitting).

Requires the test lock and report written by ``evaluate_v5_test_once.py --open-test``.
Re-reads the frozen test predictions saved by that run, checks that they reproduce the
reported per-seed metrics, and adds the pre-specified RQ1 contrast (concept + clinical
EBM minus clinical-only EBM) and the logistic-versus-EBM contrast on identical inputs,
with the same paired bootstrap. Writes data/analysis/v5_test/test_summary.json; prints
aggregate numbers only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_v5_test_once as e  # noqa: E402

CONTRASTS = {"rq1_concept_clinical_ebm_minus_clinical_only_ebm": ("concept_clinical_ebm", "clinical_only_ebm"),
             "rq1_concept_clinical_logistic_minus_clinical_only_logistic": ("concept_clinical_logistic",
                                                                            "clinical_only_logistic"),
             "concept_clinical_logistic_minus_concept_clinical_ebm": ("concept_clinical_logistic",
                                                                      "concept_clinical_ebm")}


def main():
    lock = json.loads(e.LOCK.read_text())
    report = json.loads((e.OUTPUT / "test_report.json").read_text())
    if report["test_evaluated"] is not True or report["fake_labels"]:
        raise RuntimeError("No real locked test evaluation found")
    frame = pd.read_csv(e.DATASET / "manifest.csv", low_memory=False)
    if e.sha256(e.DATASET / "manifest.csv") != lock["binding"]["manifest_sha256"]:
        raise RuntimeError("Manifest differs from the locked binding")
    schema = json.loads((e.DATASET / "feature_schema.json").read_text())
    test = np.flatnonzero(frame["split"].eq("test"))
    y = frame.loc[test, schema["outcome_column"]].to_numpy(int)
    saved = np.load(e.OUTPUT / "local_predictions.npz")
    if not np.array_equal(saved["test_rows"], test):
        raise RuntimeError("Saved prediction rows differ")
    table = pd.read_csv(e.OUTPUT / "test_metrics_by_seed.csv")
    worst = 0.0
    for row in table.itertuples():
        p = saved[f"seed{row.seed}_{row.model}_test"]
        metrics = e.outcome_metrics(y, p, row.threshold)
        worst = max(worst, *(abs(metrics[k] - getattr(row, k)) for k in metrics))
    if worst > 1e-12:
        raise RuntimeError("Saved test predictions do not reproduce the locked metrics")
    contrasts = {}
    for seed in report["seeds"]:
        for k, (name, reference) in enumerate(CONTRASTS.items()):
            contrasts[f"seed{seed}_{name}"] = e.paired(y, saved[f"seed{seed}_{reference[0]}_test"],
                                                       saved[f"seed{seed}_{reference[1]}_test"], seed + 950 + k)
    summary = {"split": "test", "patients": len(test), "deaths": int(y.sum()), "seeds": report["seeds"],
               "lock_opened_at": lock["opened_at"], "metrics_reproduced_max_abs_diff": worst,
               "additional_contrasts": contrasts}
    (e.OUTPUT / "test_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    mean = table.groupby("model")[["auroc", "auprc", "macro_f1", "brier", "logloss"]].agg(["mean", "std"])
    print(mean.round(3).to_string())
    for key, value in report["paired_differences"].items():
        print(key, {m: (round(v["delta"], 3), [round(x, 3) for x in v["ci95"]]) for m, v in value.items()
                    if m in ("auroc", "auprc")})
    for key, value in contrasts.items():
        print(key, {m: (round(v["delta"], 3), [round(x, 3) for x in v["ci95"]]) for m, v in value.items()
                    if m in ("auroc", "auprc")})


if __name__ == "__main__":
    main()
