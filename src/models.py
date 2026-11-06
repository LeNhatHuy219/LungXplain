"""
Unified model definitions and factories for MIMIC-CXR mortality prediction.
Follows dermaXAI architectural design with EfficientNet-B0 backbone,
modular diagnosis heads (mlp128 / linear), and unified exports.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
import torchvision.models as models

from src.cbm import HardJointCBM, SoftJointCBM, make_diagnosis_head
from src.ecbm import CategoricalECBM

# Aliases for compatibility
HardCBM = HardJointCBM


class BlackBoxClassifier(nn.Module):
    """
    Direct image-to-mortality classifier without concept bottleneck.
    Backbone: EfficientNet-B0 -> features (1280) -> MLP(128) diagnosis head -> 2 logits.
    """
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        head: str = "mlp128",
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        network = models.efficientnet_b0(weights=weights)
        in_features = network.classifier[1].in_features  # 1280

        self.features = nn.Sequential(network.features, network.avgpool)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = make_diagnosis_head(in_features, num_classes, head)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.dropout(self.extract_features(x))
        return self.classifier(feat)


def get_soft_joint_cbm(
    concept_names: Sequence[str],
    concept_num_classes: Dict[str, int] | None = None,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    diagnosis_head: str = "mlp128",
) -> SoftJointCBM:
    return SoftJointCBM(
        concept_names=concept_names,
        concept_num_classes=concept_num_classes,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        diagnosis_head=diagnosis_head,
    )


def get_hard_joint_cbm(
    concept_names: Sequence[str],
    concept_num_classes: Dict[str, int] | None = None,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
    diagnosis_head: str = "mlp128",
) -> HardJointCBM:
    return HardJointCBM(
        concept_names=concept_names,
        concept_num_classes=concept_num_classes,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        diagnosis_head=diagnosis_head,
    )


def get_categorical_ecbm(
    concept_names: Sequence[str],
    concept_num_classes: Dict[str, int] | None = None,
    num_classes: int = 2,
    pretrained: bool = True,
    hidden_dim: int = 32,
    dropout: float = 0.2,
) -> CategoricalECBM:
    return CategoricalECBM(
        concept_names=concept_names,
        concept_num_classes=concept_num_classes,
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
