#!/usr/bin/env python3
"""
Evaluate Concept Interventions on trained CBM and ECBM checkpoints for MIMIC-CXR.
Follows dermaXAI protocol:
- Measures diagnosis metric improvement as clinicians correct m concepts (m = 0..14).
- Computes Area Under the Intervention Curve (AUC-IC) and Delta Macro-F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cbm import HardJointCBM, SoftJointCBM
from src.dataset import CHEXPERT_14_CONCEPTS, MimicCxrConceptDataset
from src.ecbm import CategoricalECBM
from src.intervention import (
    build_mimic_concept_schema,
    compute_intervention_auc,
    evaluate_cbm_interventions,
    evaluate_ecbm_interventions,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type")
    config = checkpoint.get("config", {})

    if model_type in ("soft", "soft_cbm"):
        model = SoftJointCBM(
            concept_names=CHEXPERT_14_CONCEPTS,
            num_classes=2,
            pretrained=False,
            diagnosis_head=config.get("diagnosis_head", "mlp128"),
        )
    elif model_type in ("hard", "hard_cbm"):
        model = HardJointCBM(
            concept_names=CHEXPERT_14_CONCEPTS,
            num_classes=2,
            pretrained=False,
            diagnosis_head=config.get("diagnosis_head", "mlp128"),
        )
    elif model_type == "ecbm":
        model = CategoricalECBM(
            concept_names=CHEXPERT_14_CONCEPTS,
            num_classes=2,
            pretrained=False,
            hidden_dim=config.get("hidden_dim", 32),
        )
    else:
        raise ValueError(f"Unknown or unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    threshold = checkpoint.get("threshold", 0.5)
    return model, model_type, threshold, checkpoint


def extract_evaluation_cache(model, model_type, loader, device):
    """Passes test set through the model once to cache features, predictions, and ground-truth."""
    print("Extracting test representation cache...", flush=True)
    all_labels, all_concepts = [], []
    all_features = []
    all_cbm_vectors = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            concepts = batch["concept_indices"].cpu().numpy()

            all_labels.extend(labels)
            all_concepts.append(concepts)

            if model_type == "ecbm":
                feat = model.extract_features(images)
                all_features.append(feat)
            else:
                _, _, c_vec = model(images)
                all_cbm_vectors.append(c_vec)

    all_labels = np.asarray(all_labels, dtype=int)
    all_concepts = np.concatenate(all_concepts, axis=0)

    if model_type == "ecbm":
        features_tensor = torch.cat(all_features, dim=0)
        return features_tensor, all_concepts, all_labels
    else:
        cbm_vectors_tensor = torch.cat(all_cbm_vectors, dim=0)
        return cbm_vectors_tensor, all_concepts, all_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained best.pt")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "data/mimic_cxr/manifest_mortality.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mimic_cxr")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--strategy", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--num-random-trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    model, model_type, threshold, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    schema = build_mimic_concept_schema(CHEXPERT_14_CONCEPTS)

    dataset = MimicCxrConceptDataset(
        manifest_path=args.manifest_path,
        data_root=args.data_root,
        split=args.split,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    out = args.output_dir or args.checkpoint.parent / f"intervention_{args.strategy}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {model_type} from {args.checkpoint}")
    print(f"Calibrated decision threshold: {threshold:.2f}")

    if model_type == "ecbm":
        features_tensor, true_indices, labels = extract_evaluation_cache(model, model_type, loader, device)
        results = evaluate_ecbm_interventions(
            model=model,
            features=features_tensor,
            true_indices=true_indices,
            labels=labels,
            concept_names=CHEXPERT_14_CONCEPTS,
            threshold=threshold,
            strategy=args.strategy,
            num_random_trials=args.num_random_trials,
            seed=args.seed,
        )
    else:
        cbm_vectors_tensor, true_indices, labels = extract_evaluation_cache(model, model_type, loader, device)
        results = evaluate_cbm_interventions(
            head=model.diagnosis_head,
            concept_vectors=cbm_vectors_tensor,
            true_indices=true_indices,
            labels=labels,
            schema=schema,
            threshold=threshold,
            strategy=args.strategy,
            num_random_trials=args.num_random_trials,
            seed=args.seed,
        )

    auc_ic_f1 = compute_intervention_auc(results, metric_key="macro_f1")
    auc_ic_bacc = compute_intervention_auc(results, metric_key="balanced_accuracy")
    delta_f1 = results[-1]["macro_f1"] - results[0]["macro_f1"]
    delta_bacc = results[-1]["balanced_accuracy"] - results[0]["balanced_accuracy"]

    summary = {
        "model_type": model_type,
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold": threshold,
        "strategy": args.strategy,
        "auc_ic_macro_f1": auc_ic_f1,
        "auc_ic_balanced_accuracy": auc_ic_bacc,
        "delta_macro_f1": delta_f1,
        "delta_balanced_accuracy": delta_bacc,
        "baseline_f1": results[0]["macro_f1"],
        "full_intervention_f1": results[-1]["macro_f1"],
        "curve": results,
    }

    write_json(out / "intervention_summary.json", summary)
    csv_rows = [
        {
            "m": r["m"],
            "intervention_rate": r["intervention_rate"],
            "macro_f1": r["macro_f1"],
            "balanced_accuracy": r["balanced_accuracy"],
            "accuracy": r["accuracy"],
        }
        for r in results
    ]
    write_csv(out / "intervention_curve.csv", csv_rows)

    print(f"\n--- Intervention Results ({model_type}) ---")
    print(f"Baseline F1 (m=0):          {results[0]['macro_f1']:.4f}")
    print(f"Full Intervention F1 (m=14): {results[-1]['macro_f1']:.4f}")
    print(f"Delta Macro-F1:     {delta_f1:+.4f}")
    print(f"AUC-IC (Macro-F1):  {auc_ic_f1:.4f}")
    print(f"AUC-IC (Bal-Acc):   {auc_ic_bacc:.4f}")
    print(f"Saved artifacts to {out}")


if __name__ == "__main__":
    main()
