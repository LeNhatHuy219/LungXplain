"""
Concept loss functions for MIMIC-CXR.
Provides StateWeightedCE (cross-entropy with class-balanced weights for absent vs present states)
and helper utilities for computing training state weights.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateWeightedCE(nn.Module):
    """
    Cross-Entropy loss with per-state inverse-frequency weights:
    weight[k] = N / (K * N_k).
    """
    def __init__(self, weights: Sequence[float] | torch.Tensor):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        if w.ndim != 1 or not torch.isfinite(w).all() or torch.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid state weights for StateWeightedCE")
        self.register_buffer("weights", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)


def compute_concept_state_weights(df, concept_names: Sequence[str]) -> Dict[str, List[float]]:
    """
    Computes [weight_absent, weight_present] for each of the 14 CheXpert concepts
    from the training dataframe.
    """
    total = len(df)
    weights = {}
    for name in concept_names:
        series = df[name]
        pos = int((series == "present").sum() if series.dtype == object else (series == 1).sum())
        neg = total - pos
        w_neg = total / (2.0 * max(neg, 1))
        w_pos = total / (2.0 * max(pos, 1))
        weights[name] = [float(w_neg), float(w_pos)]
    return weights


def build_concept_criteria(
    state_weights: Dict[str, List[float]],
    concept_names: Sequence[str],
    device: torch.device,
) -> Dict[str, StateWeightedCE]:
    """Builds a dictionary of StateWeightedCE modules, one for each concept."""
    return {
        name: StateWeightedCE(state_weights[name]).to(device)
        for name in concept_names
    }
