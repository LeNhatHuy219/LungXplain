"""Single, locked evaluation of the frozen v5 models on the held-out test set.

Everything evaluated here was frozen on training/validation data: CNN weights, Platt
models, ACC parameters, EBMs, baseline models and Macro-F1 thresholds. The script

1. verifies every artifact against its run's completion stamp and the local v5 manifest;
2. runs the frozen CNNs locally (CPU, fp32) on the 256-pixel cache and checks the replay
   against the saved validation predictions (Kaggle used a T4 with mixed precision);
3. computes test predictions for ACC-ECBM (4 configurations), the black-box CNN and the
   two late-fusion models, and the six analysis baselines (clinical-only, concept-only
   and concept + clinical, each with an EBM and a logistic head);
4. only with ``--open-test``: reads test outcomes, writes a lock that binds the result to
   the SHA-256 of every frozen artifact, and computes metrics, paired bootstrap
   differences, the intervention curves of the main configuration, calibration, and a
   descriptive agreement of predicted concepts with the 96 radiologist-labelled studies.

A second ``--open-test`` run with any changed artifact is refused. Output is aggregate
only, under data/analysis/v5_test/. Without ``--open-test`` no test label is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from torchvision import transforms
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
CACHE = ROOT / "data/model_cache/cxr_resized256_v5"
OUTPUT = ROOT / "data/analysis/v5_test"
LOCK = OUTPUT / "test_lock.json"
MAIN_VARIANT = "calibrated_intervention_aware"
VARIANTS = ["uncalibrated_standard_ebm", "uncalibrated_intervention_aware",
            "calibrated_standard_ebm", MAIN_VARIANT]
BLACKBOX = ["image_only_cnn", "late_fusion_logistic", "late_fusion_ebm"]
ANALYSIS = [f"{group}_{head}" for head in ("ebm", "logistic")
            for group in ("clinical_only", "concept_only", "concept_clinical")]
BOOTSTRAP = 2000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PlattCalibration:
    """Unpickling shim with the training notebook's class name and transform rule."""

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


# ---- definitions identical to the native training notebook -------------------------
def acc_transform(probabilities, alpha, weights, reference, observed):
    residual = np.where(observed, reference - probabilities, 0.0)
    updated = np.clip(probabilities + alpha * (residual @ weights), 0.0, 1.0)
    updated[observed] = reference[observed]
    return updated


def hard_correction(probabilities, reference, observed):
    return np.where(observed, reference, probabilities)


def entropy_order(probabilities, known):
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return np.argsort(-np.where(known, entropy, -np.inf), axis=1, kind="stable")


def confirmation_mask(order, known, count):
    observed = np.zeros_like(known, dtype=bool)
    if count:
        np.put_along_axis(observed, order[:, :count], True, axis=1)
    return observed & known


def concept_logloss_parts(labels, probabilities, mask):
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    losses = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    return (losses * mask).sum(axis=1), mask.sum(axis=1)


def outcome_metrics(labels, probabilities, threshold):
    return {"auroc": roc_auc_score(labels, probabilities), "auprc": average_precision_score(labels, probabilities),
            "macro_f1": f1_score(labels, probabilities >= threshold, average="macro", zero_division=0),
            "brier": brier_score_loss(labels, probabilities),
            "logloss": log_loss(labels, probabilities, labels=[0, 1])}
# ---------------------------------------------------------------------------------------


def probability_logit(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clinical_matrix(frame, schema):
    columns = schema["model_clinical_columns"]
    result = frame[columns].copy()
    for column in columns:
        if column in schema["categorical_value_columns"]:
            mapping = {v: i for i, v in enumerate(schema["categorical_values"][column])}
            result[column] = frame[column].map(mapping)
        else:
            result[column] = pd.to_numeric(result[column], errors="raise")
    return result.to_numpy(dtype=np.float64)


def baseline_inputs(raw, z, name):
    """Inputs of the analysis-notebook baselines: raw concept probabilities (logit for
    logistic) and/or the 44 clinical columns, in the notebook's column order."""
    pieces = []
    if not name.startswith("clinical_only"):
        pieces.append(probability_logit(raw) if name.endswith("logistic") else raw)
    if not name.startswith("concept_only"):
        pieces.append(z)
    return np.column_stack(pieces)


def analysis_runs(folder, native, manifest_sha, frozen):
    """Verify the six frozen analysis baselines of every seed and return their models."""
    result = {}
    for seed, run in native.items():
        folder_seed = Path(folder) / f"seed{seed}"
        comparison = json.loads((folder_seed / "comparison_validation.json").read_text())
        provenance = comparison["provenance"]
        if (provenance["seed"] != seed or provenance["manifest_sha256"] != manifest_sha or
                provenance["source_complete_sha256"] != sha256(run / "complete.json") or
                comparison["test_evaluated"] is not False):
            raise RuntimeError(f"{folder_seed} is not bound to the native seed-{seed} run")
        for name, digest in provenance["artifacts"].items():
            if sha256(folder_seed / name) != digest:
                raise RuntimeError(f"{folder_seed / name} differs from its provenance record")
            frozen[str((folder_seed / name).relative_to(ROOT))] = digest
        frozen[str((folder_seed / "comparison_validation.json").relative_to(ROOT))] = sha256(folder_seed / "comparison_validation.json")
        frozen[str((folder_seed / "models_validation.csv").relative_to(ROOT))] = sha256(folder_seed / "models_validation.csv")
        table = pd.read_csv(folder_seed / "models_validation.csv").set_index("model")
        result[seed] = {name: {"model": joblib.load(folder_seed / f"{name}.joblib"),
                               "threshold": float(table.loc[name, "threshold"]),
                               "validation": np.load(folder_seed / f"{name}_validation.npz")["probabilities"]}
                        for name in ANALYSIS}
    return result


def load_cnn(path, outputs):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, outputs)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.eval()


TRANSFORM = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def predict_cnn(model, images, rows, batch=32):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            x = torch.stack([TRANSFORM(Image.open(images[r]).convert("RGB")) for r in rows[start:start + batch]])
            outputs.append(torch.sigmoid(model(x)).numpy().astype(np.float64))
    return np.concatenate(outputs)


def completed_runs(folder, manifest_sha):
    runs = {}
    for stamp in Path(folder).rglob("complete.json"):
        run = stamp.parent
        if not run.name.startswith("seed"):
            continue
        binding = json.loads((run / "binding.json").read_text())
        if binding["manifest_sha256"] != manifest_sha or binding.get("test_evaluated") is not False:
            raise RuntimeError(f"{run} is not a test-locked run bound to the v5 manifest")
        seed = int(binding["seed"])
        if seed in runs:
            raise RuntimeError(f"Multiple runs for seed {seed} in {folder}")
        runs[seed] = run
    return dict(sorted(runs.items()))


def verify(run, names, frozen):
    stamp = json.loads((run / "complete.json").read_text())
    for name in names:
        digest = sha256(run / name)
        if stamp["artifacts"].get(name) != digest:
            raise RuntimeError(f"{run / name} differs from its completion stamp")
        frozen[str((run / name).relative_to(ROOT))] = digest


def paired(y, model, reference, seed):
    rng = np.random.default_rng(seed)
    functions = {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss,
                 "logloss": lambda t, p: log_loss(t, p, labels=[0, 1])}
    draws = {k: [] for k in functions}
    for _ in range(BOOTSTRAP):
        rows = rng.integers(0, len(y), len(y))
        if np.unique(y[rows]).size < 2:
            continue
        for k, f in functions.items():
            draws[k].append(f(y[rows], model[rows]) - f(y[rows], reference[rows]))
    return {k: {"delta": float(f(y, model) - f(y, reference)),
                "ci95": np.quantile(draws[k], [0.025, 0.975]).tolist()} for k, f in functions.items()}


def calibration(y, p, bins=10):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y).coef_[0, 0]
    intercept = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(logit + intercept)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        intercept += step
        if abs(step) < 1e-10:
            break
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return {"calibration_intercept": float(intercept), "calibration_slope": float(slope),
            "reliability": [{"patients": int((index == b).sum()), "mean_predicted": float(p[index == b].mean()),
                             "observed_rate": float(y[index == b].mean())}
                            for b in range(len(edges) - 1) if (index == b).any()]}


def intervention_curves(ebm, z, probabilities, targets, known, y, threshold, calibrator, seed):
    order = entropy_order(probabilities, known)
    records, intervals = [], {}
    for count in range(len(targets[0]) + 1):
        observed = confirmation_mask(order, known, count)
        remaining = known & ~observed
        versions = {"no_correction": probabilities,
                    "hard_correction": hard_correction(probabilities, targets, observed),
                    "acc": acc_transform(probabilities, calibrator["alpha"], calibrator["weights"], targets, observed)}
        outcomes, parts = {}, {}
        for strategy, corrected in versions.items():
            outcomes[strategy] = ebm.predict_proba(np.column_stack([corrected, z]))[:, 1]
            parts[strategy] = concept_logloss_parts(targets, corrected, remaining)
            total = parts[strategy][1].sum()
            records.append({"requested_concepts": count, "strategy": strategy,
                            "mean_actual_confirmations": float(observed.sum(axis=1).mean()),
                            "remaining_concept_logloss": parts[strategy][0].sum() / total if total else None,
                            **outcome_metrics(y, outcomes[strategy], threshold)})
        if count in (1, 2, 14):
            rng = np.random.default_rng(seed + count)
            for reference in ("hard_correction", "no_correction"):
                draws = {"outcome_auroc": [], "outcome_logloss": [], "remaining_concept_logloss": []}
                for _ in range(BOOTSTRAP):
                    rows = rng.integers(0, len(y), len(y))
                    if np.unique(y[rows]).size == 2:
                        draws["outcome_auroc"].append(roc_auc_score(y[rows], outcomes["acc"][rows]) -
                                                      roc_auc_score(y[rows], outcomes[reference][rows]))
                    draws["outcome_logloss"].append(log_loss(y[rows], outcomes["acc"][rows], labels=[0, 1]) -
                                                    log_loss(y[rows], outcomes[reference][rows], labels=[0, 1]))
                    n = parts[reference][1][rows].sum()
                    if n:
                        draws["remaining_concept_logloss"].append(
                            (parts["acc"][0][rows].sum() - parts[reference][0][rows].sum()) / n)
                n = parts[reference][1].sum()
                point = {"outcome_auroc": roc_auc_score(y, outcomes["acc"]) - roc_auc_score(y, outcomes[reference]),
                         "outcome_logloss": log_loss(y, outcomes["acc"], labels=[0, 1]) -
                                            log_loss(y, outcomes[reference], labels=[0, 1]),
                         "remaining_concept_logloss": (parts["acc"][0].sum() - parts[reference][0].sum()) / n if n else None}
                intervals[f"m{count}_acc_minus_{reference}"] = {
                    k: {"delta": point[k], "ci95": np.quantile(v, [0.025, 0.975]).tolist() if v else None}
                    for k, v in draws.items()}
    return records, intervals


def radiologist_agreement(frame, rows, probabilities, concepts):
    labels = pd.read_csv(DATASET / "radiologist_labels_test.csv")
    index = {d: i for i, d in enumerate(frame.loc[rows, "dicom_id"])}
    result = []
    for concept in concepts:
        if concept not in labels.columns:
            continue
        subset = labels[labels[concept].isin([0, 1])]
        positions = [index[d] for d in subset["dicom_id"] if d in index]
        truth = subset[subset["dicom_id"].isin(index)][concept].to_numpy(int)
        entry = {"concept": concept, "studies": len(truth), "positives": int(truth.sum())}
        if len(truth) and np.unique(truth).size == 2:
            entry["auroc"] = float(roc_auc_score(truth, probabilities[positions, concepts.index(concept)]))
        result.append(entry)
    return result


def main(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha256(DATASET / "manifest.csv")
    schema = json.loads((DATASET / "feature_schema.json").read_text())
    frame = pd.read_csv(DATASET / "manifest.csv", low_memory=False)
    concepts = list(schema["concept_label_columns"])
    valid = np.flatnonzero(frame["split"].eq("valid"))
    test = np.flatnonzero(frame["split"].eq("test"))
    z = clinical_matrix(frame, schema)
    targets = frame[concepts].to_numpy(np.float64)
    known = frame[["known_" + c for c in concepts]].to_numpy() == 1
    index = json.loads((CACHE / "index.json").read_text())
    if index["configuration"]["manifest_sha256"] != manifest_sha or index["rows"] != len(frame):
        raise RuntimeError("Pixel cache is not bound to the v5 manifest")
    images = [CACHE / item["file"] for item in index["images"]]
    for row in np.r_[valid, test]:
        if sha256(images[row]) != index["images"][row]["sha256"]:
            raise RuntimeError("Pixel cache integrity failed")
    torch.set_num_threads(args.threads)
    frozen, predictions, replay, thresholds, extras = {}, {}, {}, {}, {}
    replay_metrics, local = {}, {}
    y_valid = frame.loc[valid, schema["outcome_column"]].to_numpy(int)

    def record(key, saved, replayed, outcome=True):
        """Replay deviation of the local CPU predictions from the saved Kaggle predictions;
        for outcome models also the change of validation metrics it causes."""
        replay[f"{key}_valid_max_abs_diff"] = float(np.max(np.abs(saved - replayed)))
        if outcome:
            replay_metrics[key] = {
                "auroc_local_minus_saved": float(roc_auc_score(y_valid, replayed) - roc_auc_score(y_valid, saved)),
                "auprc_local_minus_saved": float(average_precision_score(y_valid, replayed) -
                                                 average_precision_score(y_valid, saved)),
                "brier_local_minus_saved": float(brier_score_loss(y_valid, replayed) - brier_score_loss(y_valid, saved)),
                "mean_abs_diff": float(np.mean(np.abs(saved - replayed)))}
    native = completed_runs(args.native, manifest_sha)
    blackbox = completed_runs(args.blackbox, manifest_sha)
    baselines = analysis_runs(args.analysis, native, manifest_sha, frozen)
    seeds = sorted(set(native) & set(blackbox))
    if not seeds or seeds != sorted(native) or seeds != sorted(blackbox):
        raise RuntimeError("Native and black-box runs must cover the same seeds")
    for seed in seeds:
        run = native[seed]
        names = ["final_cnn/best.pt", "concept_predictions.npz", "platt.joblib", "calibrated_acc.json",
                 "uncalibrated_acc.json"]
        names += [f"{v}/{f}" for v in VARIANTS for f in ("ebm.joblib", "threshold_validation.json",
                                                          "outcome_validation.npz")]
        verify(run, names, frozen)
        model = load_cnn(run / "final_cnn/best.pt", len(concepts))
        raw = np.full((len(frame), len(concepts)), np.nan)
        raw[valid] = predict_cnn(model, images, valid)
        raw[test] = predict_cnn(model, images, test)
        del model
        with np.load(run / "concept_predictions.npz") as saved:
            record(f"seed{seed}_concept", saved["probabilities"][valid], raw[valid], outcome=False)
        platt = load_platt(run / "platt.joblib")
        calibrated = np.full_like(raw, np.nan)
        calibrated[valid], calibrated[test] = platt.transform(raw[valid]), platt.transform(raw[test])
        for variant in VARIANTS:
            ebm = joblib.load(run / variant / "ebm.joblib")
            source = calibrated if variant.startswith("calibrated") else raw
            with np.load(run / variant / "outcome_validation.npz") as saved:
                local_valid = ebm.predict_proba(np.column_stack([source[valid], z[valid]]))[:, 1]
                record(f"seed{seed}_{variant}", saved["probabilities"], local_valid)
            local[f"seed{seed}_acc_ecbm_{variant}_valid"] = local_valid
            predictions[(seed, f"acc_ecbm_{variant}")] = ebm.predict_proba(np.column_stack([source[test], z[test]]))[:, 1]
            thresholds[(seed, f"acc_ecbm_{variant}")] = json.loads((run / variant / "threshold_validation.json").read_text())["threshold"]
        for name, entry in baselines[seed].items():
            local_valid = entry["model"].predict_proba(baseline_inputs(raw[valid], z[valid], name))[:, 1]
            record(f"seed{seed}_{name}", entry["validation"], local_valid)
            local[f"seed{seed}_{name}_valid"] = local_valid
            predictions[(seed, name)] = entry["model"].predict_proba(baseline_inputs(raw[test], z[test], name))[:, 1]
            thresholds[(seed, name)] = entry["threshold"]
        acc = json.loads((run / "calibrated_acc.json").read_text())
        extras[seed] = {"calibrated_test": calibrated[test], "acc": {"alpha": float(acc["alpha"]),
                                                                     "weights": np.asarray(acc["weights"])},
                        "main_ebm": joblib.load(run / MAIN_VARIANT / "ebm.joblib")}
        run = blackbox[seed]
        verify(run, ["final_cnn/best.pt", "image_mortality_predictions.npz", "late_fusion_ebm.joblib",
                     "late_fusion_logistic.joblib", "validation_summary.json", "outcome_validation.npz"], frozen)
        model = load_cnn(run / "final_cnn/best.pt", 1)
        image = np.full(len(frame), np.nan)
        image[valid] = predict_cnn(model, images, valid)[:, 0]
        image[test] = predict_cnn(model, images, test)[:, 0]
        del model
        with np.load(run / "image_mortality_predictions.npz") as saved:
            record(f"seed{seed}_blackbox", saved["probabilities"][valid], image[valid])
        local[f"seed{seed}_image_only_cnn_valid"] = image[valid]
        fusion = np.column_stack([probability_logit(image[test]), z[test]])
        fusion_valid = np.column_stack([probability_logit(image[valid]), z[valid]])
        predictions[(seed, "image_only_cnn")] = image[test]
        with np.load(run / "outcome_validation.npz") as saved:
            for name in ("late_fusion_ebm", "late_fusion_logistic"):
                fusion_model = joblib.load(run / f"{name}.joblib")
                local_valid = fusion_model.predict_proba(fusion_valid)[:, 1]
                record(f"seed{seed}_{name}", saved[name], local_valid)
                local[f"seed{seed}_{name}_valid"] = local_valid
                predictions[(seed, name)] = fusion_model.predict_proba(fusion)[:, 1]
        for row in json.loads((run / "validation_summary.json").read_text()):
            thresholds[(seed, row["variant"])] = row["threshold"]
    status = {"seeds": seeds, "frozen_artifacts": len(frozen), "replay": replay,
              "replay_validation_metric_change": replay_metrics,
              "test_predictions": len(predictions), "test_labels_read": False}
    np.savez_compressed(OUTPUT / "local_predictions.npz", valid_rows=valid, test_rows=test, **local,
                        **{f"seed{seed}_{name}_test": p for (seed, name), p in predictions.items()})
    (OUTPUT / "pre_open_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: v for k, v in status.items() if k not in ("replay", "replay_validation_metric_change")}))
    print(json.dumps({"max_abs_auroc_change_from_replay":
                      max(abs(v["auroc_local_minus_saved"]) for v in replay_metrics.values())}))
    print(json.dumps({"max_replay_abs_diff": max(replay.values())}))
    if args.fake_labels:
        # Exercises the complete metrics branch with random outcomes and concept labels;
        # no real test label is read, no lock is written, nothing is printed but status.
        rng = np.random.default_rng(0)
        y = rng.permutation(np.r_[np.ones(50, int), np.zeros(len(test) - 50, int)])
        targets = targets.copy()
        known = known.copy()
        targets[test] = rng.integers(0, 2, targets[test].shape)
        known[test] = rng.random(known[test].shape) < 0.3
        output = OUTPUT / "fake_label_check"
    elif args.open_test:
        binding = {"frozen_artifacts": frozen, "manifest_sha256": manifest_sha}
        script = sha256(Path(__file__))
        if LOCK.exists():
            lock = json.loads(LOCK.read_text())
            if lock["binding"] != binding:
                raise RuntimeError("Test was already opened for a different set of frozen artifacts")
            lock["script_sha256_runs"].append(script)
        else:
            lock = {"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "binding": binding,
                    "script_sha256_runs": [script]}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        y = frame.loc[test, schema["outcome_column"]].to_numpy(int)
        output = OUTPUT
    else:
        print("Test labels not read. Re-run with --open-test to perform the single test evaluation.")
        return
    output.mkdir(parents=True, exist_ok=True)
    rows, comparisons, curves, intervals, calibration_rows, radiologist = [], {}, [], {}, {}, {}
    for (seed, name), p in predictions.items():
        rows.append({"seed": seed, "model": name, "threshold": thresholds[(seed, name)],
                     **outcome_metrics(y, p, thresholds[(seed, name)])})
    main_name = f"acc_ecbm_{MAIN_VARIANT}"
    for seed in seeds:
        reference = predictions[(seed, main_name)]
        for k, name in enumerate(BLACKBOX + ANALYSIS + [f"acc_ecbm_{v}" for v in VARIANTS if v != MAIN_VARIANT]):
            comparisons[f"seed{seed}_acc_ecbm_minus_{name}"] = paired(y, reference, predictions[(seed, name)], seed + 900 + k)
        records, ci = intervention_curves(extras[seed]["main_ebm"], z[test], extras[seed]["calibrated_test"],
                                          targets[test], known[test], y, thresholds[(seed, main_name)],
                                          extras[seed]["acc"], seed)
        curves += [{"seed": seed, **r} for r in records]
        intervals[f"seed{seed}"] = ci
        calibration_rows[f"seed{seed}"] = calibration(y, reference)
        if not args.fake_labels:
            radiologist[f"seed{seed}"] = radiologist_agreement(frame, test, extras[seed]["calibrated_test"], concepts)
    table = pd.DataFrame(rows)
    table.to_csv(output / "test_metrics_by_seed.csv", index=False)
    summary = table.groupby("model")[["auroc", "auprc", "macro_f1", "brier", "logloss"]].agg(["mean", "std"])
    summary.to_csv(output / "test_metrics_summary.csv")
    pd.DataFrame(curves).to_csv(output / "test_interventions_main_variant.csv", index=False)
    report = {"split": "test", "test_evaluated": not args.fake_labels, "fake_labels": args.fake_labels, "seeds": seeds, "patients": len(test),
              "deaths": int(y.sum()), "main_variant": MAIN_VARIANT, "bootstrap_repeats": BOOTSTRAP,
              "replay": replay, "replay_validation_metric_change": replay_metrics,
              "paired_differences": comparisons, "intervention_intervals": intervals,
              "calibration_main_variant": calibration_rows, "radiologist_concept_agreement": radiologist}
    (output / "test_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    if args.fake_labels:
        print(json.dumps({"fake_label_branch_completed": True, "rows": len(table), "curve_rows": len(curves)}))
    else:
        print(summary.round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native", type=Path, default=ROOT / "data/kaggle_v5_output/native_output")
    parser.add_argument("--blackbox", type=Path, default=ROOT / "data/kaggle_v5_output/blackbox_output")
    parser.add_argument("--analysis", type=Path, default=ROOT / "data/kaggle_v5_output/analysis_output/lungxplain_analysis")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--open-test", action="store_true",
                        help="Read test outcomes once; binds the result to all frozen artifact hashes")
    parser.add_argument("--fake-labels", action="store_true",
                        help="Exercise the metrics branch with random labels; reads no real test label")
    arguments = parser.parse_args()
    if arguments.open_test and arguments.fake_labels:
        parser.error("--open-test and --fake-labels are exclusive")
    main(arguments)
