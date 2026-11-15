"""RQ2 on validation: ACC-ECBM (main variant, no intervention) minus black-box baselines.

Reads the saved validation predictions of the downloaded native and black-box runs
(``data/kaggle_v5_output/{native,blackbox}_output``); both runs must be bound to the
local v5 manifest and complete. Paired bootstrap over the same validation patients.
Test rows are never read. Output is aggregate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
MAIN_VARIANT = "calibrated_intervention_aware"
BASELINES = ["image_only_cnn", "late_fusion_logistic", "late_fusion_ebm"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1])}


def paired(y, model, reference, seed, repeats):
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("auroc", "auprc", "brier", "logloss")}
    for _ in range(repeats):
        rows = rng.integers(0, len(y), len(y))
        if np.unique(y[rows]).size < 2:
            continue
        a, b = metrics(y[rows], model[rows]), metrics(y[rows], reference[rows])
        for k in draws:
            draws[k].append(a[k] - b[k])
    a, b = metrics(y, model), metrics(y, reference)
    return {k: {"delta": a[k] - b[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in draws.items()}


def runs(folder):
    found = {}
    for stamp in folder.rglob("complete.json"):
        run = stamp.parent
        if run.name.startswith("seed"):
            binding = json.loads((run / "binding.json").read_text())
            if binding["manifest_sha256"] != sha256(DATASET / "manifest.csv"):
                raise RuntimeError(f"{run} is not bound to the local v5 manifest")
            seed = int(binding["seed"])
            if seed in found:
                raise RuntimeError(f"Multiple runs for seed {seed} under {folder}")
            found[seed] = run
    return found


def verified(run, name):
    stamp = json.loads((run / "complete.json").read_text())
    if stamp["artifacts"].get(name) != sha256(run / name):
        raise RuntimeError(f"{run / name} differs from its completion stamp")
    return np.load(run / name)


def main(args):
    output = args.output.resolve()
    output.relative_to(ROOT / "data")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATASET / "manifest.csv", low_memory=False, usecols=["split", "diagnosis_binary"])
    valid = np.flatnonzero(frame["split"].eq("valid"))
    y = frame.loc[valid, "diagnosis_binary"].to_numpy(int)
    native, blackbox = runs(args.native), runs(args.blackbox)
    seeds = sorted(set(native) & set(blackbox))
    if not seeds:
        raise RuntimeError("No seed has both a complete native and a complete black-box run")
    rows, report = [], {"split": "valid", "test_evaluated": False, "seeds": seeds,
                        "reference": f"ACC-ECBM {MAIN_VARIANT}, no intervention", "per_seed": {}}
    for seed in seeds:
        with verified(native[seed], f"{MAIN_VARIANT}/outcome_validation.npz") as saved:
            if not np.array_equal(saved["rows"], valid):
                raise RuntimeError("Native validation rows misaligned")
            acc_ecbm = saved["probabilities"]
        with verified(blackbox[seed], "outcome_validation.npz") as saved:
            if not np.array_equal(saved["rows"], valid):
                raise RuntimeError("Black-box validation rows misaligned")
            predictions = {name: saved[name] for name in BASELINES}
        rows.append({"seed": seed, "model": "acc_ecbm", **metrics(y, acc_ecbm)})
        report["per_seed"][seed] = {}
        for k, name in enumerate(BASELINES):
            rows.append({"seed": seed, "model": name, **metrics(y, predictions[name])})
            report["per_seed"][seed][f"acc_ecbm_minus_{name}"] = paired(
                y, acc_ecbm, predictions[name], seed + 700 + k, args.bootstrap)
    table = pd.DataFrame(rows)
    table.to_csv(output / "rq2_validation_by_seed.csv", index=False)
    summary = table.groupby("model")[["auroc", "auprc", "brier", "logloss"]].agg(["mean", "std"])
    summary.to_csv(output / "rq2_validation_summary.csv")
    (output / "rq2_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.round(4).to_string())
    for seed in seeds:
        print(json.dumps({"seed": seed, **{k: v["auroc"] for k, v in report["per_seed"][seed].items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=ROOT / "data/kaggle_v5_output/native_output")
    parser.add_argument("--blackbox", type=Path, default=ROOT / "data/kaggle_v5_output/blackbox_output")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis/v5_rq2")
    parser.add_argument("--bootstrap", type=int, default=2000)
    main(parser.parse_args())
