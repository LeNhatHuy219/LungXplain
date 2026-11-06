"""
Concept Bottleneck Models (CBM) for MIMIC-CXR following dermaXAI architecture:
- SoftJointCBM (Soft CBM): Continuous concept probabilities fed into diagnosis head.
- HardJointCBM (Hard CBM): Categorical one-hot bottleneck.
Adapted for 14 CheXpert clinical concepts and binary in-hospital mortality target.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_diagnosis_head(
    input_dim: int,
    num_classes: int = 2,
    architecture: str = "mlp128",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Constructs diagnosis head receiving concept bottleneck representation.
    Matches dermaXAI architecture:
      - 'linear': Linear(input_dim, num_classes)
      - 'mlp128': Linear -> LayerNorm -> ReLU -> Dropout -> Linear
    """
    if architecture == "linear":
        return nn.Linear(input_dim, num_classes)
    if architecture == "mlp128":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    raise ValueError(f"Unknown diagnosis head '{architecture}'; expected 'linear' or 'mlp128'")


def hard_categorical(logits: torch.Tensor, straight_through: bool = False) -> torch.Tensor:
    """
    Converts concept logits to one-hot hard categorical vector.
    Straight-through: forward pass uses hard one-hot argmax, backward pass propagates gradient through softmax.
    """
    probabilities = logits.softmax(dim=-1)
    hard = F.one_hot(probabilities.argmax(dim=-1), logits.shape[-1]).to(logits.dtype)
    return hard + (probabilities - probabilities.detach()) if straight_through else hard


class SoftJointCBM(nn.Module):
    """
    Soft Concept Bottleneck Model:
    Image X -> EfficientNet-B0 -> 14 Concept Heads (each 2-class softmax) -> 28D Soft Vector -> Diagnosis Head -> Y
    """
    def __init__(
        self,
        concept_names: Sequence[str],
        concept_num_classes: Dict[str, int] | None = None,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        diagnosis_head: str = "mlp128",
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.concept_num_classes = {
            name: (concept_num_classes[name] if concept_num_classes else 2)
            for name in self.concept_names
        }
        self.total_concept_states = sum(self.concept_num_classes[name] for name in self.concept_names)
        self.num_classes = num_classes

        # Backbone: EfficientNet-B0 (1280 features)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280

        self.features = nn.Sequential(backbone.features, backbone.avgpool)
        self.dropout = nn.Dropout(p=dropout)

        # 14 concept heads (each mapping 1280 -> 2)
        self.concept_heads = nn.ModuleDict({
            name: nn.Linear(in_features, self.concept_num_classes[name])
            for name in self.concept_names
        })

        # Diagnosis head receiving the concatenated concept probabilities (28 dimensions)
        self.diagnosis_head = make_diagnosis_head(self.total_concept_states, num_classes, diagnosis_head)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        feat = self.dropout(self.extract_features(x))

        # Raw concept logits per head (B, 2)
        concept_logits: Dict[str, torch.Tensor] = {
            name: self.concept_heads[name](feat) for name in self.concept_names
        }

        # Softmax per concept group -> concatenated 28-dimensional soft vector
        concept_vector = torch.cat(
            [torch.softmax(concept_logits[name], dim=-1) for name in self.concept_names],
            dim=-1,
        )

        diag_logits = self.diagnosis_head(concept_vector)
        return diag_logits, concept_logits, concept_vector


class HardJointCBM(SoftJointCBM):
    """
    Hard Concept Bottleneck Model (Hard CBM):
    Bottleneck one-hot categorical representation trained jointly with straight-through estimator.
    Matches dermaXAI M4 / Hard CBM definition.
    """
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features = self.dropout(self.extract_features(x))
        concept_logits = {name: self.concept_heads[name](features) for name in self.concept_names}

        # One-hot concept bottleneck (28 dimensions for 14 binary concepts)
        vector = torch.cat(
            [hard_categorical(concept_logits[name], self.training)
             for name in self.concept_names],
            dim=-1,
        )
        diag_logits = self.diagnosis_head(vector)
        return diag_logits, concept_logits, vector
