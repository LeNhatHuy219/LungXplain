#!/usr/bin/env python3
"""
Aggregate test metrics across multiple seeds (42, 123, 2026) for MIMIC-CXR mortality models.
Computes Mean ± SD for Mortality and Concept metrics, formatting directly for thesis tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results/mimic_cxr"
DEFAULT_SEEDS = [42, 123, 2026]
MODELS = ["blackbox", "soft", "hard", "ecbm"]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def format_ms(mean: float, std: float, percent: bool = False) -> str:
    if percent:
        return f"{mean * 100:.2f} ± {std * 100:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_model(model_name: str, base_dir: Path, seeds: list[int]) -> dict:
    records = []
    for seed in seeds:
        # Search for folder with pattern seed{seed}
        candidates = list(base_dir.glob(f"*seed{seed}*"))
        if not candidates:
            continue
        # Take the most recently modified candidate if multiple exist
        best_cand = max(candidates, key=lambda p: p.stat().st_mtime)
        metrics_file = best_cand / "test_metrics.json"
        if not metrics_file.exists():
            continue
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            records.append({"seed": seed, "data": data, "path": str(metrics_file)})
        except Exception:
            continue

    if not records:
        return {"model": model_name, "n_seeds": 0, "records": []}

    # Extract target metrics at validation-tuned threshold
    f1_list, bacc_list, roc_auc_list, pr_auc_list = [], [], [], []
    c_f1_list, emr_list = [], []

    for r in records:
        m = r["data"].get("test_at_validation_threshold") or r["data"].get("test_y_at_validation_threshold", {})
        if "macro_f1" in m:
            f1_list.append(m["macro_f1"])
        if "balanced_accuracy" in m:
            bacc_list.append(m["balanced_accuracy"])
        if "roc_auc" in m and m["roc_auc"] is not None:
            roc_auc_list.append(m["roc_auc"])
        if "pr_auc" in m and m["pr_auc"] is not None:
            pr_auc_list.append(m["pr_auc"])

        # Concept metrics if available
        cm = r["data"].get("test_concept_metrics", {})
        if "mean_concept_macro_f1" in cm:
            c_f1_list.append(cm["mean_concept_macro_f1"])
        if "exact_match_ratio" in cm:
            emr_list.append(cm["exact_match_ratio"])

    f1_m, f1_s = mean_std(f1_list)
    bacc_m, bacc_s = mean_std(bacc_list)
    roc_m, roc_s = mean_std(roc_auc_list)
    pr_m, pr_s = mean_std(pr_auc_list)
    cf1_m, cf1_s = mean_std(c_f1_list)
    emr_m, emr_s = mean_std(emr_list)

    return {
        "model": model_name,
        "n_seeds": len(records),
        "available_seeds": [r["seed"] for r in records],
        "macro_f1": {"mean": f1_m, "std": f1_s, "str": format_ms(f1_m, f1_s)},
        "balanced_acc": {"mean": bacc_m, "std": bacc_s, "str": format_ms(bacc_m, bacc_s)},
        "roc_auc": {"mean": roc_m, "std": roc_s, "str": format_ms(roc_m, roc_s)},
        "pr_auc": {"mean": pr_m, "std": pr_s, "str": format_ms(pr_m, pr_s)},
        "concept_macro_f1": {"mean": cf1_m, "std": cf1_s, "str": format_ms(cf1_m, cf1_s) if c_f1_list else "N/A"},
        "exact_match_ratio": {"mean": emr_m, "std": emr_s, "str": format_ms(emr_m, emr_s, percent=True) if emr_list else "N/A"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-json", type=Path, default=RESULTS_ROOT / "multi_seed_benchmark.json")
    args = parser.parse_args()

    results = []
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY ACROSS SEEDS: {args.seeds}")
    print(f"{'='*80}")

    for model in MODELS:
        model_dir = args.results_dir / model
        if not model_dir.exists():
            continue
        agg = aggregate_model(model, model_dir, args.seeds)
        results.append(agg)

    # Print Table
    header = f"| {'Model':<18} | {'Seeds':<8} | {'Macro-F1':<17} | {'Balanced Acc':<17} | {'ROC-AUC':<17} | {'Concept F1':<17} |"
    print(header)
    print(f"|{'-'*20}|{'-'*10}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*19}|")

    for r in results:
        m_name = r["model"].upper()
        s_count = f"{r['n_seeds']}/{len(args.seeds)}"
        f1_str = r["macro_f1"]["str"]
        bacc_str = r["balanced_acc"]["str"]
        roc_str = r["roc_auc"]["str"]
        c_str = r["concept_macro_f1"]["str"]
        print(f"| {m_name:<18} | {s_count:<8} | {f1_str:<17} | {bacc_str:<17} | {roc_str:<17} | {c_str:<17} |")

    print(f"{'='*80}\n")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved aggregated benchmark to {args.output_json}")


if __name__ == "__main__":
    main()
