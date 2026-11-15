"""Validation-only sensitivity and calibration analyses for the v5 native ACC-ECBM run.

Runs locally on CPU after the private native training output has been downloaded under
data/. Uses only train/validation outcomes; test rows are never scored. Steps per seed:

1. Bind the run to the local v5 manifest (SHA-256 in ``binding.json``) and to its
   completion stamp (SHA-256 of every artifact used here).
2. Rebuild the main variant's inputs: per-concept Platt scaling refitted on the saved
   out-of-fold training probabilities, the saved ACC parameters and the saved
   intervention-aware training mask. Replay check: the saved Kaggle EBM applied to the
   rebuilt validation inputs must reproduce the saved validation probabilities.
3. Refit a reference EBM on this machine with the run's configuration, and the
   sensitivity EBMs with the same configuration and seed:
   - temporal: laboratory values linked only by exact ``hadm_id``
     (``manifest_temporal_sensitivity.csv``);
   - no_triage: the 7 triage values and their 7 missingness flags removed.
   Paired bootstrap differences are computed against the local reference, so every
   compared pair is fitted on the same machine.
4. Calibration (intercept, slope, Brier, quantile reliability) and AP/PA strata of the
   saved main-model validation predictions.

Outputs are aggregate only: ``<output>/sensitivity_validation.json`` and CSV tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
MAIN_VARIANT = "calibrated_intervention_aware"
TRIAGE_PREFIX = "clinical_triage_"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clinical_matrix(frame, schema, columns):
    result = frame[columns].copy()
    for column in columns:
        if column in schema["categorical_value_columns"]:
            mapping = {v: i for i, v in enumerate(schema["categorical_values"][column])}
            if not frame[column].dropna().isin(mapping).all():
                raise ValueError("Unknown clinical category")
            result[column] = frame[column].map(mapping)
        else:
            result[column] = pd.to_numeric(result[column], errors="raise")
    return result.to_numpy(dtype=np.float64)


def feature_types(schema, concepts, columns):
    return ["continuous"] * len(concepts) + [
        "nominal" if c in schema["categorical_value_columns"] or c.endswith("_missing") else "continuous"
        for c in columns]


class PlattCalibration:
    """Unpickling shim with the notebook's class name; transform is the notebook's rule."""

    def transform(self, probabilities):
        result = np.array(probabilities, dtype=float, copy=True)
        for j, model in enumerate(self.models):
            if model is not None:
                p = np.clip(probabilities[:, j], 1e-6, 1 - 1e-6)
                result[:, j] = model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return result


def load_platt(path):
    import __main__
    __main__.PlattCalibration = PlattCalibration
    return joblib.load(path)


def platt_refit(raw_train, labels, supervised, raw_apply):
    """Same rule as the notebook's PlattCalibration: logistic on the logit, C=1."""
    result = np.array(raw_apply, dtype=float, copy=True)
    for j in range(raw_train.shape[1]):
        eligible = supervised[:, j]
        if np.unique(labels[eligible, j]).size != 2:
            continue
        p = np.clip(raw_train[eligible, j], 1e-6, 1 - 1e-6)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), labels[eligible, j])
        q = np.clip(raw_apply[:, j], 1e-6, 1 - 1e-6)
        result[:, j] = model.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]
    return result


def acc_transform(probabilities, alpha, weights, reference, observed):
    residual = np.where(observed, reference - probabilities, 0.0)
    updated = np.clip(probabilities + alpha * (residual @ weights), 0.0, 1.0)
    updated[observed] = reference[observed]
    return updated


def metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1])}


def calibration(y, p, bins=10):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope_model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    # Calibration-in-the-large: intercept with the logit as a fixed offset (Newton steps).
    intercept = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(logit[:, 0] + intercept)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        intercept += step
        if abs(step) < 1e-10:
            break
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    reliability = [{"bin": int(b), "patients": int((index == b).sum()),
                    "mean_predicted": float(p[index == b].mean()), "observed_rate": float(y[index == b].mean())}
                   for b in range(len(edges) - 1) if (index == b).any()]
    return {"calibration_intercept": float(intercept), "calibration_slope": float(slope_model.coef_[0, 0]),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()), "reliability": reliability}


def paired_bootstrap(y, model, reference, seed, repeats):
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in ("auroc", "auprc", "brier", "logloss")}
    for _ in range(repeats):
        rows = rng.integers(0, len(y), len(y))
        if np.unique(y[rows]).size < 2:
            continue
        a, b = metrics(y[rows], model[rows]), metrics(y[rows], reference[rows])
        for name in samples:
            samples[name].append(a[name] - b[name])
    point_a, point_b = metrics(y, model), metrics(y, reference)
    return {name: {"delta": point_a[name] - point_b[name],
                   "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                   "valid_resamples": len(v)} for name, v in samples.items()}


def fit_ebm(inputs, y, names, types, config, seed):
    model = ExplainableBoostingClassifier(feature_names=names, feature_types=types, random_state=seed, **config)
    return model.fit(inputs, y)


def analyze_seed(run, frame, sensitivity, schema, args):
    binding = json.loads((run / "binding.json").read_text())
    stamp = json.loads((run / "complete.json").read_text())
    seed = int(binding["seed"])
    if binding["manifest_sha256"] != sha256(DATASET / "manifest.csv"):
        raise RuntimeError("Run is not bound to the local v5 manifest")
    used = ["concept_predictions.npz", "platt.joblib", "calibrated_acc.json", f"{MAIN_VARIANT}/ebm.joblib",
            f"{MAIN_VARIANT}/outcome_validation.npz", f"{MAIN_VARIANT}/training_intervention_mask.npz"]
    for name in used:
        if stamp["artifacts"].get(name) != sha256(run / name):
            raise RuntimeError(f"Artifact differs from its completion stamp: {name}")
    if stamp.get("test_evaluated") is not False:
        raise RuntimeError("Unexpected test evaluation flag")
    concepts = list(schema["concept_label_columns"])
    clinical = list(schema["model_clinical_columns"])
    train = np.flatnonzero(frame["split"].eq("train"))
    valid = np.flatnonzero(frame["split"].eq("valid"))
    test = np.flatnonzero(frame["split"].eq("test"))
    y_train = frame.loc[train, schema["outcome_column"]].to_numpy(int)
    y_valid = frame.loc[valid, schema["outcome_column"]].to_numpy(int)
    targets = frame[concepts].to_numpy(np.float64)
    supervised = frame[["ignore_" + c for c in concepts]].to_numpy() == 0
    with np.load(run / "concept_predictions.npz") as saved:
        raw = saved["probabilities"]
        if not (np.array_equal(saved["train_rows"], train) and np.array_equal(saved["valid_rows"], valid)
                and np.isnan(raw[test]).all() and np.isfinite(raw[np.r_[train, valid]]).all()):
            raise RuntimeError("Concept prediction alignment or test isolation failed")
    platt = load_platt(run / "platt.joblib")
    calibrated_train = platt.transform(raw[train])
    calibrated_valid = platt.transform(raw[valid])
    refit_valid = platt_refit(raw[train], targets[train].astype(np.float32), supervised[train], raw[valid])
    platt_refit_difference = float(np.max(np.abs(refit_valid - calibrated_valid)))
    acc = json.loads((run / "calibrated_acc.json").read_text())
    with np.load(run / f"{MAIN_VARIANT}/training_intervention_mask.npz") as saved:
        mask = saved["observed"].astype(bool)
    known = frame[["known_" + c for c in concepts]].to_numpy() == 1
    if mask.shape != (len(train), len(concepts)) or (mask & ~known[train]).any():
        raise RuntimeError("Training intervention mask is inconsistent with explicit references")
    train_concepts = acc_transform(calibrated_train, float(acc["alpha"]), np.asarray(acc["weights"]),
                                   targets[train], mask)
    z = clinical_matrix(frame, schema, clinical)
    z_temporal = clinical_matrix(sensitivity, schema, clinical)
    kaggle_ebm = joblib.load(run / f"{MAIN_VARIANT}/ebm.joblib")
    with np.load(run / f"{MAIN_VARIANT}/outcome_validation.npz") as saved:
        saved_valid = saved["probabilities"]
    replay = kaggle_ebm.predict_proba(np.column_stack([calibrated_valid, z[valid]]))[:, 1]
    replay_error = float(np.max(np.abs(replay - saved_valid)))
    if replay_error > args.replay_tolerance:
        raise RuntimeError(f"Replay of saved validation predictions failed (max error {replay_error:.3g})")
    config = binding["configuration"]["ebm"]
    names = concepts + clinical
    types = feature_types(schema, concepts, clinical)
    reference = fit_ebm(np.column_stack([train_concepts, z[train]]), y_train, names, types, config, seed)
    p_reference = reference.predict_proba(np.column_stack([calibrated_valid, z[valid]]))[:, 1]
    temporal = fit_ebm(np.column_stack([train_concepts, z_temporal[train]]), y_train, names, types, config, seed)
    p_temporal = temporal.predict_proba(np.column_stack([calibrated_valid, z_temporal[valid]]))[:, 1]
    keep = [i for i, c in enumerate(clinical) if not c.startswith(TRIAGE_PREFIX)]
    kept = [clinical[i] for i in keep]
    if len(clinical) - len(kept) != 14:
        raise RuntimeError("Expected 7 triage values and 7 triage missingness flags")
    no_triage = fit_ebm(np.column_stack([train_concepts, z[train][:, keep]]), y_train, concepts + kept,
                        feature_types(schema, concepts, kept), config, seed)
    p_no_triage = no_triage.predict_proba(np.column_stack([calibrated_valid, z[valid][:, keep]]))[:, 1]
    views = frame.loc[valid, "ViewPosition"].to_numpy()
    strata = {}
    for view in ("AP", "PA"):
        rows = views == view
        strata[view] = {"patients": int(rows.sum()), "deaths": int(y_valid[rows].sum()),
                        **(metrics(y_valid[rows], saved_valid[rows]) if np.unique(y_valid[rows]).size == 2 else {})}
    return {
        "seed": seed,
        "replay_max_abs_error": replay_error,
        "platt_local_refit_max_abs_diff": platt_refit_difference,
        "local_reference_vs_kaggle_max_abs_diff": float(np.max(np.abs(p_reference - saved_valid))),
        "kaggle_main_model": metrics(y_valid, saved_valid),
        "local_reference": metrics(y_valid, p_reference),
        "temporal_exact_hadm_only": metrics(y_valid, p_temporal),
        "no_triage": metrics(y_valid, p_no_triage),
        "temporal_minus_reference": paired_bootstrap(y_valid, p_temporal, p_reference, seed + 501, args.bootstrap),
        "no_triage_minus_reference": paired_bootstrap(y_valid, p_no_triage, p_reference, seed + 502, args.bootstrap),
        "calibration_kaggle_main_model": calibration(y_valid, saved_valid),
        "view_strata_kaggle_main_model": strata,
    }


def main(args):
    output = args.output.resolve()
    output.relative_to(ROOT / "data")
    output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((DATASET / "feature_schema.json").read_text())
    validation = json.loads((DATASET / "validation_report.json").read_text())
    if validation.get("passed") is not True:
        raise RuntimeError("A passing dataset validation is required")
    frame = pd.read_csv(DATASET / "manifest.csv", low_memory=False)
    sensitivity = pd.read_csv(DATASET / "manifest_temporal_sensitivity.csv", low_memory=False)
    if not (frame["dicom_id"].equals(sensitivity["dicom_id"]) and frame["split"].equals(sensitivity["split"])):
        raise RuntimeError("Sensitivity manifest is not row-aligned with the primary manifest")
    runs = sorted(p.parent for p in args.runs.resolve().rglob("complete.json") if p.parent.name.startswith("seed"))
    if not runs:
        raise RuntimeError("No completed native seed runs found")
    results = [analyze_seed(run, frame, sensitivity, schema, args) for run in runs]
    rows = []
    for result in results:
        for model in ("kaggle_main_model", "local_reference", "temporal_exact_hadm_only", "no_triage"):
            rows.append({"seed": result["seed"], "model": model, **result[model]})
    table = pd.DataFrame(rows)
    table.to_csv(output / "sensitivity_validation_by_seed.csv", index=False)
    summary = table.groupby("model")[["auroc", "auprc", "brier", "logloss"]].agg(["mean", "std"])
    summary.to_csv(output / "sensitivity_validation_summary.csv")
    report = {"split": "valid", "test_evaluated": False, "seeds": [r["seed"] for r in results],
              "main_variant": MAIN_VARIANT, "bootstrap_repeats": args.bootstrap,
              "manifest_sha256": sha256(DATASET / "manifest.csv"),
              "sensitivity_manifest_sha256": sha256(DATASET / "manifest_temporal_sensitivity.csv"),
              "per_seed": results}
    (output / "sensitivity_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.round(4).to_string())
    for result in results:
        print(json.dumps({"seed": result["seed"], "replay_max_abs_error": result["replay_max_abs_error"],
                          "temporal_delta_auroc": result["temporal_minus_reference"]["auroc"],
                          "no_triage_delta_auroc": result["no_triage_minus_reference"]["auroc"],
                          "calibration_slope": result["calibration_kaggle_main_model"]["calibration_slope"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="Folder containing seed<seed>/ native run folders")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis/v5_sensitivity")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--replay-tolerance", type=float, default=1e-8)
    main(parser.parse_args())
