"""
Concept Intervention evaluation module for MIMIC-CXR.
Follows dermaXAI protocol:
- Replays frozen diagnosis head under simulated clinician corrections.
- Supports both Soft CBM and Hard CBM (via one-hot concept vectors).
- Supports ECBM via exact clamped inference.
- Evaluates intervention curve m = 0..14 (m: number of intervened concepts).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.metrics import binary_metrics


def build_mimic_concept_schema(concept_names: Sequence[str]) -> Dict[str, Any]:
    names = list(concept_names)
    num_classes = {name: 2 for name in names}
    offsets = {name: 2 * i for i, name in enumerate(names)}
    return {
        "concept_names": names,
        "concept_num_classes": num_classes,
        "total_states": len(names) * 2,
        "offsets": offsets,
    }


@torch.no_grad()
def evaluate_cbm_interventions(
    head: nn.Module,
    concept_vectors: torch.Tensor,
    true_indices: np.ndarray,
    labels: np.ndarray,
    schema: Dict[str, Any],
    threshold: float,
    strategy: str = "sequential",  # 'sequential' or 'random'
    num_random_trials: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates concept intervention on a trained CBM diagnosis head.
    concept_vectors: [N, 28] predicted concept probabilities or one-hot vectors.
    true_indices: [N, 14] ground-truth concept indices (0 or 1).
    labels: [N] ground-truth mortality labels (0 or 1).
    """
    head.eval()
    device = next(head.parameters()).device
    names = schema["concept_names"]
    count = len(names)
    vectors = torch.as_tensor(concept_vectors, dtype=torch.float32, device=device)
    true_indices = np.asarray(true_indices, dtype=int)
    labels = np.asarray(labels, dtype=int)

    rng = np.random.default_rng(seed)
    results = []

    # Baseline (m = 0): no intervention
    baseline_probs = head(vectors).softmax(-1)[:, 1].cpu().numpy()
    baseline_metrics = binary_metrics(labels, baseline_probs, threshold=threshold)
    results.append({
        "m": 0,
        "intervention_rate": 0.0,
        "macro_f1": baseline_metrics["macro_f1"],
        "balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "accuracy": baseline_metrics["accuracy"],
        "roc_auc": baseline_metrics["roc_auc"],
        "intervened_concepts": [],
    })

    for m in range(1, count + 1):
        if strategy == "sequential":
            # Intervene on the first m concepts
            subsets = [list(range(m))]
        elif strategy == "random":
            # Sample random subsets of size m
            all_possible = list(itertools.combinations(range(count), m))
            n_select = min(num_random_trials, len(all_possible))
            indices = rng.choice(len(all_possible), size=n_select, replace=False)
            subsets = [list(all_possible[idx]) for idx in indices]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        trial_metrics = []
        for subset in subsets:
            corrected = vectors.clone()
            for group_idx in subset:
                c_name = names[group_idx]
                start = schema["offsets"][c_name]
                k = schema["concept_num_classes"][c_name]
                targets = torch.as_tensor(true_indices[:, group_idx], dtype=torch.long, device=device)
                corrected[:, start:start + k] = nn.functional.one_hot(targets, k).to(corrected.dtype)

            probs = head(corrected).softmax(-1)[:, 1].cpu().numpy()
            m_metrics = binary_metrics(labels, probs, threshold=threshold)
            trial_metrics.append(m_metrics)

        # Average metrics across trials for subset size m:
        avg_f1 = float(np.mean([item["macro_f1"] for item in trial_metrics]))
        avg_bacc = float(np.mean([item["balanced_accuracy"] for item in trial_metrics]))
        avg_acc = float(np.mean([item["accuracy"] for item in trial_metrics]))

        results.append({
            "m": m,
            "intervention_rate": m / count,
            "macro_f1": avg_f1,
            "balanced_accuracy": avg_bacc,
            "accuracy": avg_acc,
            "intervened_concepts": [names[idx] for idx in subsets[0]] if strategy == "sequential" else "random_sample",
        })

    return results


@torch.no_grad()
def evaluate_ecbm_interventions(
    model: nn.Module,
    features: torch.Tensor,
    true_indices: np.ndarray,
    labels: np.ndarray,
    concept_names: Sequence[str],
    threshold: float,
    strategy: str = "sequential",
    num_random_trials: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates concept intervention on ECBM via exact clamped inference.
    features: [N, 1280] precomputed backbone features.
    true_indices: [N, 14] ground-truth concept indices.
    labels: [N] ground-truth mortality labels.
    """
    from src.ecbm import exact_inference

    model.eval()
    device = features.device
    names = list(concept_names)
    count = len(names)
    true_indices = np.asarray(true_indices, dtype=int)
    labels = np.asarray(labels, dtype=int)

    rng = np.random.default_rng(seed)
    results = []

    # Baseline (m = 0): no clamp
    out_base = exact_inference(model, features)
    baseline_probs = out_base["class_probabilities"][:, 1].cpu().numpy()
    baseline_metrics = binary_metrics(labels, baseline_probs, threshold=threshold)
    results.append({
        "m": 0,
        "intervention_rate": 0.0,
        "macro_f1": baseline_metrics["macro_f1"],
        "balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "accuracy": baseline_metrics["accuracy"],
        "roc_auc": baseline_metrics["roc_auc"],
        "intervened_concepts": [],
    })

    for m in range(1, count + 1):
        if strategy == "sequential":
            subsets = [list(range(m))]
        elif strategy == "random":
            all_possible = list(itertools.combinations(range(count), m))
            n_select = min(num_random_trials, len(all_possible))
            indices = rng.choice(len(all_possible), size=n_select, replace=False)
            subsets = [list(all_possible[idx]) for idx in indices]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        trial_metrics = []
        for subset in subsets:
            clamp = {
                group_idx: torch.as_tensor(true_indices[:, group_idx], dtype=torch.long, device=device)
                for group_idx in subset
            }
            out_int = exact_inference(model, features, clamp=clamp)
            probs = out_int["class_probabilities"][:, 1].cpu().numpy()
            m_metrics = binary_metrics(labels, probs, threshold=threshold)
            trial_metrics.append(m_metrics)

        avg_f1 = float(np.mean([item["macro_f1"] for item in trial_metrics]))
        avg_bacc = float(np.mean([item["balanced_accuracy"] for item in trial_metrics]))
        avg_acc = float(np.mean([item["accuracy"] for item in trial_metrics]))

        results.append({
            "m": m,
            "intervention_rate": m / count,
            "macro_f1": avg_f1,
            "balanced_accuracy": avg_bacc,
            "accuracy": avg_acc,
            "intervened_concepts": [names[idx] for idx in subsets[0]] if strategy == "sequential" else "random_sample",
        })

    return results


def compute_intervention_auc(results: List[Dict[str, Any]], metric_key: str = "macro_f1") -> float:
    """Computes Area Under the Intervention Curve (AUC-IC) normalized to [0, 1]."""
    rates = [row["intervention_rate"] for row in results]
    values = [row[metric_key] for row in results]
    return float(np.trapz(values, rates))
