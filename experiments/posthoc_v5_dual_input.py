"""POST HOC exploratory analysis, requested after the v5 test set was opened.

Design (written before fitting): experiments/V5_POSTHOC_DUAL_INPUT.md. One variant only: the
EBM receives the 14 automated (Platt) concept probabilities, the 14 post-intervention concept
values (ACC, supervisor formula and saved alpha/W unchanged) and the 44 clinical inputs. It is
compared with the main configuration refitted on the same machine.

Default stage (validation): per seed, refit both EBMs locally with the saved Platt model, ACC
parameters and training intervention mask; choose each Macro-F1 threshold on validation;
evaluate intervention curves and the pre-specified paired comparisons; freeze the EBMs by hash.
``--test``: requires the freeze, replays the frozen CNN on the test images (CPU, as in the
locked evaluation), checks the replay against the locked test predictions, then reads test
outcomes once. Output is aggregate only, under data/analysis/v5_posthoc_dual_input/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_v5_test_once as e  # noqa: E402

OUTPUT = e.ROOT / "data/analysis/v5_posthoc_dual_input"
FREEZE = OUTPUT / "freeze.json"
LOCK = OUTPUT / "posthoc_test_lock.json"
NATIVE = e.ROOT / "data/kaggle_v5_output/native_output/lungxplain_native/runs"
SEEDS = [42, 123, 2026]
EBM = {"interactions": 10, "max_rounds": 1500, "outer_bags": 4, "max_bins": 64,
       "max_interaction_bins": 16, "early_stopping_rounds": 50, "n_jobs": 4}  # native CFG["ebm"]
MODELS = ("reference", "dual")


def threshold(y, p):
    from sklearn.metrics import f1_score
    candidates = np.unique(np.r_[0.0, p, np.nextafter(1.0, 2.0)])
    scores = [f1_score(y, p >= t, average="macro", zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


class Data:
    def __init__(self):
        self.schema = json.loads((e.DATASET / "feature_schema.json").read_text())
        self.frame = pd.read_csv(e.DATASET / "manifest.csv", low_memory=False)
        self.manifest_sha = e.sha256(e.DATASET / "manifest.csv")
        self.concepts = list(self.schema["concept_label_columns"])
        clinical = list(self.schema["model_clinical_columns"])
        split = self.frame["split"]
        self.train, self.valid, self.test = (np.flatnonzero(split.eq(s)) for s in ("train", "valid", "test"))
        self.targets = self.frame[self.concepts].to_numpy(np.float32)  # as in the native notebook
        self.known = self.frame[["known_" + c for c in self.concepts]].to_numpy() == 1
        self.z = e.clinical_matrix(self.frame, self.schema)
        clinical_types = ["nominal" if c in self.schema["categorical_value_columns"] or c.endswith("_missing")
                          else "continuous" for c in clinical]
        self.names = {"reference": self.concepts + clinical,
                      "dual": self.concepts + [c + " (after intervention)" for c in self.concepts] + clinical}
        self.types = {"reference": ["continuous"] * 14 + clinical_types,
                      "dual": ["continuous"] * 28 + clinical_types}

    def outcome(self, rows):
        return self.frame.loc[rows, self.schema["outcome_column"]].to_numpy(int)


def inputs(kind, automated, post, z):
    return np.column_stack([post, z] if kind == "reference" else [automated, post, z])


def predict(model, kind, automated, post, z):
    return model.predict_proba(inputs(kind, automated, post, z))[:, 1]


def seed_parameters(run, data):
    frozen = {}
    e.verify(run, ["concept_predictions.npz", "platt.joblib", "calibrated_acc.json",
                   f"{e.MAIN_VARIANT}/training_intervention_mask.npz", f"{e.MAIN_VARIANT}/outcome_validation.npz",
                   f"{e.MAIN_VARIANT}/ebm.joblib", "final_cnn/best.pt"], frozen)
    acc = json.loads((run / "calibrated_acc.json").read_text())
    return {"platt": e.load_platt(run / "platt.joblib"), "frozen": frozen,
            "acc": {"alpha": float(acc["alpha"]), "weights": np.asarray(acc["weights"])},
            "mask": np.load(run / f"{e.MAIN_VARIANT}/training_intervention_mask.npz")["observed"].astype(bool)}


def evaluate(data, models, thresholds, automated, rows, y, acc, seed):
    """Intervention curves (m = 0..14) for both EBMs and the pre-specified paired comparisons."""
    targets, known, z = data.targets[rows], data.known[rows], data.z[rows]
    order = e.entropy_order(automated, known)
    records, outcomes = [], {}
    for count in range(len(data.concepts) + 1):
        observed = e.confirmation_mask(order, known, count)
        versions = {"automated": automated, "hard": e.hard_correction(automated, targets, observed),
                    "acc": e.acc_transform(automated, acc["alpha"], acc["weights"], targets, observed)}
        for kind in MODELS:
            for strategy, post in versions.items():
                p = predict(models[kind], kind, automated, post, z)
                outcomes[(kind, strategy, count)] = p
                records.append({"seed": seed, "model": kind, "strategy": strategy, "requested_concepts": count,
                                "mean_actual_confirmations": float(observed.sum(axis=1).mean()),
                                **e.outcome_metrics(y, p, thresholds[kind])})
    comparisons = {"m0_dual_minus_reference": e.paired(y, outcomes[("dual", "automated", 0)],
                                                        outcomes[("reference", "automated", 0)], seed + 700)}
    for count in (1, 2):
        dual_acc = outcomes[("dual", "acc", count)]
        pairs = {"a_dual_acc_minus_dual_automated": outcomes[("dual", "automated", 0)],
                 "b_dual_acc_minus_reference_acc": outcomes[("reference", "acc", count)],
                 "c_dual_acc_minus_dual_hard": outcomes[("dual", "hard", count)]}
        for k, (name, reference) in enumerate(pairs.items()):
            comparisons[f"m{count}_{name}"] = e.paired(y, dual_acc, reference, seed + 710 + 10 * count + k)
        comparisons[f"m{count}_reference_acc_minus_reference_automated"] = e.paired(
            y, outcomes[("reference", "acc", count)], outcomes[("reference", "automated", 0)], seed + 740 + count)
    return records, comparisons


def stage_validation(data):
    if FREEZE.exists():
        raise RuntimeError("Validation stage already frozen; use a new output directory for any change")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    y_train, y_valid = data.outcome(data.train), data.outcome(data.valid)
    curves, comparisons, freeze, replay = [], {}, {"seeds": {}, "manifest_sha256": data.manifest_sha}, {}
    for seed in SEEDS:
        run = NATIVE / f"seed{seed}"
        params = seed_parameters(run, data)
        raw = np.load(run / "concept_predictions.npz")["probabilities"]
        auto_train, auto_valid = params["platt"].transform(raw[data.train]), params["platt"].transform(raw[data.valid])
        post_train = e.acc_transform(auto_train, params["acc"]["alpha"], params["acc"]["weights"],
                                     data.targets[data.train], params["mask"])
        folder = OUTPUT / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        models, thresholds = {}, {}
        for kind in MODELS:
            model = ExplainableBoostingClassifier(feature_names=data.names[kind], feature_types=data.types[kind],
                                                  random_state=seed, **EBM)
            model.fit(inputs(kind, auto_train, post_train, data.z[data.train]), y_train)
            models[kind] = model
            thresholds[kind] = threshold(y_valid, predict(model, kind, auto_valid, auto_valid, data.z[data.valid]))
            joblib.dump(model, folder / f"{kind}_ebm.joblib")
        saved = np.load(run / f"{e.MAIN_VARIANT}/outcome_validation.npz")["probabilities"]
        local = predict(models["reference"], "reference", auto_valid, auto_valid, data.z[data.valid])
        replay[seed] = {"reference_local_minus_kaggle_max_abs": float(np.abs(local - saved).max()),
                        "reference_local_minus_kaggle_auroc": float(roc_auc_score(y_valid, local) -
                                                                    roc_auc_score(y_valid, saved))}
        (folder / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        records, comparisons[f"seed{seed}"] = evaluate(data, models, thresholds, auto_valid, data.valid,
                                                       y_valid, params["acc"], seed)
        curves += records
        freeze["seeds"][seed] = {"artifacts": {n: e.sha256(folder / n) for n in
                                               ("reference_ebm.joblib", "dual_ebm.joblib", "thresholds.json")},
                                 "source_artifacts": params["frozen"]}
    pd.DataFrame(curves).to_csv(OUTPUT / "validation_curves.csv", index=False)
    report = {"split": "valid", "post_hoc": True, "replay_reference_vs_kaggle": replay, "comparisons": comparisons}
    (OUTPUT / "validation_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    freeze.update({"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script_sha256": e.sha256(Path(__file__)),
                   "test_evaluated": False})
    FREEZE.write_text(json.dumps(freeze, indent=2) + "\n")
    summarize(pd.DataFrame(curves), comparisons, replay)


def stage_test(data, threads):
    if not FREEZE.exists():
        raise RuntimeError("Run and freeze the validation stage first")
    if LOCK.exists():
        raise RuntimeError("The post hoc test evaluation was already performed once")
    freeze = json.loads(FREEZE.read_text())
    if freeze["manifest_sha256"] != data.manifest_sha:
        raise RuntimeError("Manifest differs from the frozen validation stage")
    torch.set_num_threads(threads)
    index = json.loads((e.CACHE / "index.json").read_text())
    images = [e.CACHE / item["file"] for item in index["images"]]
    for row in data.test:
        if e.sha256(images[row]) != index["images"][row]["sha256"]:
            raise RuntimeError("Pixel cache integrity failed")
    locked = np.load(e.OUTPUT / "local_predictions.npz")
    prepared = {}
    for seed in SEEDS:
        run, folder = NATIVE / f"seed{seed}", OUTPUT / f"seed{seed}"
        entry = freeze["seeds"][str(seed)]
        for name, digest in entry["artifacts"].items():
            if e.sha256(folder / name) != digest:
                raise RuntimeError(f"{folder / name} changed after the validation freeze")
        params = seed_parameters(run, data)
        if params["frozen"] != entry["source_artifacts"]:
            raise RuntimeError("Source artifacts differ from the validation freeze")
        cnn = e.load_cnn(run / "final_cnn/best.pt", len(data.concepts))
        automated = params["platt"].transform(e.predict_cnn(cnn, images, data.test))
        del cnn
        main = joblib.load(run / f"{e.MAIN_VARIANT}/ebm.joblib").predict_proba(
            np.column_stack([automated, data.z[data.test]]))[:, 1]
        difference = float(np.abs(main - locked[f"seed{seed}_acc_ecbm_{e.MAIN_VARIANT}_test"]).max())
        if difference > 1e-12:
            raise RuntimeError("CNN replay does not reproduce the locked test predictions")
        prepared[seed] = {"automated": automated, "acc": params["acc"],
                          "models": {k: joblib.load(folder / f"{k}_ebm.joblib") for k in MODELS},
                          "thresholds": json.loads((folder / "thresholds.json").read_text())}
    LOCK.write_text(json.dumps({"opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "post_hoc": True,
                                "freeze_sha256": e.sha256(FREEZE), "script_sha256": e.sha256(Path(__file__))},
                               indent=2) + "\n")
    y = data.outcome(data.test)
    curves, comparisons = [], {}
    for seed, item in prepared.items():
        records, comparisons[f"seed{seed}"] = evaluate(data, item["models"], item["thresholds"], item["automated"],
                                                       data.test, y, item["acc"], seed)
        curves += records
    pd.DataFrame(curves).to_csv(OUTPUT / "test_curves_posthoc.csv", index=False)
    report = {"split": "test", "post_hoc": True, "exploratory": True, "patients": len(y), "deaths": int(y.sum()),
              "comparisons": comparisons}
    (OUTPUT / "test_report_posthoc.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    summarize(pd.DataFrame(curves), comparisons, None)


def summarize(curves, comparisons, replay):
    if replay:
        print(json.dumps({"reference_replay_vs_kaggle": replay}))
    table = curves[curves.requested_concepts.isin([0, 1, 2, 5])]
    mean = table.groupby(["model", "strategy", "requested_concepts"])[["auroc", "auprc", "logloss"]].mean()
    print(mean.round(4).to_string())
    for seed, items in comparisons.items():
        for name, value in items.items():
            print(seed, name, {m: (round(value[m]["delta"], 4), [round(x, 4) for x in value[m]["ci95"]])
                               for m in ("auroc", "logloss")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Post hoc single test evaluation of the frozen EBMs")
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    dataset = Data()
    stage_test(dataset, arguments.threads) if arguments.test else stage_validation(dataset)
