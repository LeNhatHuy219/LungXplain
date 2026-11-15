"""Independent local audit of the completed private v5 black-box run (RQ2 baselines).

Checks every seed of ``data/kaggle_v5_output/blackbox_output`` against the local v5
manifest and the native ACC-ECBM run of the same seed: artifact hashes, dataset and
protocol binding, test isolation, outer/inner split reconstruction, checkpoint selection
from the saved histories (as implemented and under the declared early-stopping rule), the
refit budget, late-fusion inputs and configuration, prediction replay and validation
metrics. Writes data/audits/v5_blackbox_run/audit.json; prints aggregate results only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
OUTPUT = ROOT / "data/audits/v5_blackbox_run"
SKIPPED = ("last.pt", ".zip", ".partial")  # not downloaded on purpose
CNN_KEYS = ["outer_folds", "inner_fraction", "max_epochs", "min_epochs", "patience", "batch_size",
            "image_size", "backbone_lr", "head_lr", "weight_decay", "gradient_clip", "pos_weight_cap"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def threshold(y, p):
    candidates = np.unique(np.r_[0.0, p, np.nextafter(1.0, 2.0)])
    scores = [f1_score(y, p >= t, average="macro", zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def selected_epoch(history, declared, cfg):
    best, epoch, stale = -math.inf, 0, 0
    for entry in history:
        if entry["inner_mortality_auroc"] > best + 1e-8:
            best, epoch, stale = entry["inner_mortality_auroc"], entry["epoch"], 0
        else:
            stale += 1
        if declared and entry["epoch"] >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    return epoch


def main(args):
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    manifest_sha = sha256(DATASET / "manifest.csv")
    schema_sha = sha256(DATASET / "feature_schema.json")
    schema = json.loads((DATASET / "feature_schema.json").read_text())
    frame = pd.read_csv(DATASET / "manifest.csv", low_memory=False)
    clinical = schema["model_clinical_columns"]
    train = np.flatnonzero(frame["split"].eq("train"))
    valid = np.flatnonzero(frame["split"].eq("valid"))
    test = np.flatnonzero(frame["split"].eq("test"))
    y = frame[schema["outcome_column"]].to_numpy(float)
    z = frame[clinical].copy()
    for column in clinical:
        if column in schema["categorical_value_columns"]:
            z[column] = frame[column].map({v: i for i, v in enumerate(schema["categorical_values"][column])})
        else:
            z[column] = pd.to_numeric(z[column], errors="raise")
    z = z.to_numpy(float)
    names = ["image_mortality_logit"] + clinical
    root = args.runs / "lungxplain_blackbox/runs"
    report = {"seeds": {}, "manifest_sha256": manifest_sha}
    for seed_dir in sorted(root.glob("seed*")):
        seed = int(seed_dir.name.removeprefix("seed"))
        s = f"seed{seed}_"
        stamp = json.loads((seed_dir / "complete.json").read_text())
        present = [n for n in stamp["artifacts"] if (seed_dir / n).is_file()]
        missing = [n for n in stamp["artifacts"] if not (seed_dir / n).is_file()]
        check(s + "artifact_hashes", all(sha256(seed_dir / n) == stamp["artifacts"][n] for n in present))
        check(s + "only_intentionally_skipped_missing",
              all(n.endswith(SKIPPED) or (n.startswith("fold") and n.endswith("best.pt")) for n in missing))
        binding = json.loads((seed_dir / "binding.json").read_text())
        cfg = binding["configuration"]
        native = json.loads((args.native / f"seed{seed}/binding.json").read_text()) \
            if (args.native / f"seed{seed}/binding.json").exists() else \
            json.loads(next(args.native.rglob(f"seed{seed}/binding.json")).read_text())
        check(s + "dataset_binding", binding["manifest_sha256"] == manifest_sha and
              binding["schema_sha256"] == schema_sha and binding["seed"] == seed)
        check(s + "test_not_evaluated", binding["test_evaluated"] is False and stamp.get("test_evaluated", False) is False)
        check(s + "same_cnn_protocol_as_native",
              all(cfg[k] == native["configuration"][k] for k in CNN_KEYS))
        check(s + "same_ebm_config_as_native", cfg["ebm"] == native["configuration"]["ebm"])
        check(s + "same_cache_and_imagenet_weights",
              binding["cache_index_sha256"] == native["cache_index_sha256"] and
              binding["imagenet_sha256"] == native["imagenet_sha256"])
        check(s + "same_environment_as_native", binding["environment"] == native["environment"])
        saved = np.load(seed_dir / "image_mortality_predictions.npz")
        image, folds = saved["probabilities"], saved["fold_id"]
        check(s + "row_alignment", np.array_equal(saved["train_rows"], train) and np.array_equal(saved["valid_rows"], valid))
        check(s + "test_predictions_absent", np.isnan(image[test]).all())
        seen = image[np.r_[train, valid]]
        check(s + "train_valid_probabilities_valid", np.isfinite(seen).all() and ((seen >= 0) & (seen <= 1)).all())
        selected_actual, selected_declared, split_ok = [], [], True
        outer = StratifiedKFold(cfg["outer_folds"], shuffle=True, random_state=seed)
        for fold, (development, heldout) in enumerate(outer.split(train, y[train])):
            split_ok &= np.array_equal(np.sort(train[heldout]), train[folds[train] == fold])
            spec = json.loads((seed_dir / f"fold{fold}/stage.json").read_text())
            fitting, inner = train_test_split(train[development], test_size=cfg["inner_fraction"],
                                              random_state=seed + fold, stratify=y[train[development]])
            split_ok &= (spec["fitting_rows"] == fitting.tolist() and spec["selection_rows"] == inner.tolist()
                         and spec["stage_seed"] == seed + fold
                         and not np.intersect1d(train[heldout], np.r_[fitting, inner]).size)
            history = json.loads((seed_dir / f"fold{fold}/history.json").read_text())
            split_ok &= [h["epoch"] for h in history] == list(range(1, len(history) + 1))
            selected_actual.append(selected_epoch(history, False, cfg))
            selected_declared.append(selected_epoch(history, True, cfg))
        check(s + "outer_inner_splits_reconstructed", split_ok and (folds[np.r_[valid, test]] == -1).all())
        summary = json.loads((seed_dir / "cnn_summary.json").read_text())
        check(s + "selected_epochs_match_history", [f["best_epoch"] for f in summary["folds"]] == selected_actual)
        budget = max(1, int(np.median(selected_actual)))
        final_spec = json.loads((seed_dir / "final_cnn/stage.json").read_text())
        check(s + "refit_protocol", summary["refit_epochs"] == budget and summary["test_predicted"] is False and
              final_spec["fitting_rows"] == train.tolist() and final_spec["selection_rows"] == [] and
              final_spec["stage_seed"] == seed + 100 and final_spec["refit_epochs"] == budget and
              len(json.loads((seed_dir / "final_cnn/history.json").read_text())) == budget)
        ebm = joblib.load(seed_dir / "late_fusion_ebm.joblib")
        logistic = joblib.load(seed_dir / "late_fusion_logistic.joblib")
        selection = json.loads((seed_dir / "late_fusion_logistic_selection.json").read_text())
        params = ebm.get_params()
        check(s + "late_fusion_inputs", list(ebm.feature_names_in_) == names and ebm.n_features_in_ == 45 and
              logistic.n_features_in_ == 45)
        check(s + "late_fusion_config", all(params[k] == v for k, v in cfg["ebm"].items()) and
              params["random_state"] == seed and
              logistic.named_steps["logistic"].C == selection["best_c"] and
              selection["selection_split"] == "train_cv")
        fusion_valid = np.column_stack([logit(image[valid]), z[valid]])
        outcome = np.load(seed_dir / "outcome_validation.npz")
        replay = {"image_only_cnn": image[valid],
                  "late_fusion_ebm": ebm.predict_proba(fusion_valid)[:, 1],
                  "late_fusion_logistic": logistic.predict_proba(fusion_valid)[:, 1]}
        diffs = {k: float(np.max(np.abs(outcome[k] - v))) for k, v in replay.items()}
        check(s + "prediction_replay", np.array_equal(outcome["rows"], valid) and max(diffs.values()) <= 1e-12)
        yv = y[valid].astype(int)
        metrics_ok = True
        for row in json.loads((seed_dir / "validation_summary.json").read_text()):
            p = outcome[row["variant"]]
            metrics_ok &= (abs(row["threshold"] - threshold(yv, p)) < 1e-12 and
                           abs(row["auroc"] - roc_auc_score(yv, p)) < 1e-12 and
                           abs(row["auprc"] - average_precision_score(yv, p)) < 1e-12 and
                           abs(row["brier"] - brier_score_loss(yv, p)) < 1e-12 and
                           abs(row["logloss"] - log_loss(yv, p, labels=[0, 1])) < 1e-12)
        check(s + "validation_metrics_recomputed", metrics_ok)
        report["seeds"][seed] = {"selected_epochs": selected_actual,
                                 "selected_epochs_declared_rule": selected_declared,
                                 "refit_epochs": budget,
                                 "refit_epochs_declared_rule": max(1, int(np.median(selected_declared))),
                                 "fold_epochs_run": [len(json.loads((seed_dir / f"fold{f}/history.json").read_text()))
                                                     for f in range(cfg["outer_folds"])],
                                 "logistic_best_c": selection["best_c"], "replay_max_abs_diff": diffs,
                                 "artifacts_checked": len(present), "artifacts_not_downloaded": len(missing)}
    check("three_seeds", sorted(report["seeds"]) == [42, 123, 2026])
    report["checks"] = checks
    report["passed"] = all(checks.values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({"checks": len(checks), "passed": report["passed"],
                      "failed": [k for k, v in checks.items() if not v]}))
    for seed, info in report["seeds"].items():
        print(json.dumps({"seed": seed, **{k: info[k] for k in ("selected_epochs", "selected_epochs_declared_rule",
                                                                 "refit_epochs", "refit_epochs_declared_rule",
                                                                 "fold_epochs_run", "logistic_best_c")}}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "data/kaggle_v5_output/blackbox_output")
    parser.add_argument("--native", type=Path, default=ROOT / "data/kaggle_v5_output/native_output")
    main(parser.parse_args())
