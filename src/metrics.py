"""Binary metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score,
)


def binary_metrics(targets, probabilities, threshold: float = 0.5) -> dict:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.ndim != 1 or targets.shape != probabilities.shape or not len(targets):
        raise ValueError("Targets and probabilities must be aligned nonempty vectors")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(targets, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both_classes = len(np.unique(targets)) == 2
    pr_auc = None
    if both_classes:
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        pr_auc = float(auc(recall, precision))
    return {
        "n": len(targets), "positive": int(targets.sum()), "negative": int((targets == 0).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else None,
        "roc_auc": float(roc_auc_score(targets, probabilities)) if both_classes else None,
        "pr_auc": pr_auc,
        "average_precision": float(average_precision_score(targets, probabilities)) if both_classes else None,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def select_validation_threshold(targets, probabilities) -> tuple[float, list[dict]]:
    """Maximize balanced accuracy on 0.05..0.95; break ties nearest 0.5."""
    if set(np.asarray(targets).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both classes in validation")
    sweep = []
    for tick in range(5, 96):
        threshold = tick / 100
        metrics = binary_metrics(targets, probabilities, threshold)
        sweep.append({"threshold": threshold, "balanced_accuracy": metrics["balanced_accuracy"],
                      "macro_f1": metrics["macro_f1"]})
    best = min(sweep, key=lambda row: (-round(row["balanced_accuracy"], 12),
                                     abs(row["threshold"] - 0.5), row["threshold"]))
    return best["threshold"], sweep


def concept_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    concept_names: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluates multi-label concept predictions across 14 CheXpert concepts.
    Returns per-concept macro_f1/accuracy, overall mean macro-f1, mean accuracy,
    and exact match ratio (all 14 concepts correct).
    """
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.ndim != 2 or targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must be 2D arrays with identical shape (N, num_concepts)")
    
    n_samples, n_concepts = targets.shape
    if concept_names is None:
        concept_names = [f"concept_{i}" for i in range(n_concepts)]
    elif len(concept_names) != n_concepts:
        raise ValueError("Length of concept_names must match number of concept columns")

    predictions = (probabilities >= threshold).astype(int)
    exact_matches = np.all(predictions == targets, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    per_concept = {}
    f1_list = []
    acc_list = []

    for idx, c_name in enumerate(concept_names):
        c_target = targets[:, idx]
        c_pred = predictions[:, idx]
        c_f1 = float(f1_score(c_target, c_pred, average="macro", zero_division=0))
        c_acc = float(accuracy_score(c_target, c_pred))
        f1_list.append(c_f1)
        acc_list.append(c_acc)
        per_concept[c_name] = {
            "macro_f1": c_f1,
            "accuracy": c_acc,
            "positives": int(c_target.sum()),
            "total": n_samples,
        }

    return {
        "n_samples": n_samples,
        "num_concepts": n_concepts,
        "exact_match_ratio": exact_match_ratio,
        "mean_concept_macro_f1": float(np.mean(f1_list)),
        "mean_concept_accuracy": float(np.mean(acc_list)),
        "per_concept": per_concept,
    }

