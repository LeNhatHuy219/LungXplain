"""
Categorical Energy-Based Concept Bottleneck Model (ECBM) for MIMIC-CXR.
Follows the official Xu et al. (ICLR 2024) formulation as implemented in dermaXAI:
- Three energies:
    1. E_class(x, y): Image-to-Label energy
    2. E_concept(x, c): Image-to-Concept energy
    3. E_global(c, y): Concept-to-Label energy
- Exact enumeration inference over 2^14 = 16,384 concept profiles.
- Supports concept clamping for exact intervention analysis.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _gate(hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """Multiplicative gating interaction: relu(hidden * (1.0 + normalize(embedding)))"""
    return F.relu(hidden * (1.0 + F.normalize(embedding, dim=-1, eps=1e-12)))


class CategoricalECBM(nn.Module):
    def __init__(
        self,
        concept_names: Sequence[str],
        concept_num_classes: Dict[str, int] | None = None,
        num_classes: int = 2,
        pretrained: bool = True,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.state_counts = [
            (concept_num_classes[name] if concept_num_classes else 2)
            for name in self.concept_names
        ]
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Backbone: EfficientNet-B0 (1280 features)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280
        self.features = nn.Sequential(backbone.features, backbone.avgpool)
        self.dropout = nn.Dropout(p=dropout)

        # Class embedding and concept state embeddings initialized ~ N(0, 1)
        self.class_embedding = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.concept_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(count, hidden_dim)) for count in self.state_counts
        ])

        # Energy projection networks:
        # 1. E_class: x -> y
        self.class_projection = nn.Linear(in_features, hidden_dim)
        self.class_output = nn.Linear(hidden_dim, 1)

        # 2. E_concept: x -> c_g
        self.concept_projection = nn.Linear(in_features, hidden_dim)
        self.concept_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in self.state_counts])

        # 3. E_global: c -> y
        self.global_projection = nn.Linear(len(self.state_counts) * hidden_dim, hidden_dim)
        self.global_output = nn.Linear(hidden_dim, 1)

        # Complete discrete profile space (16,384 for 14 binary concepts) for exact enumeration:
        profiles = torch.tensor(
            list(itertools.product(*[range(count) for count in self.state_counts])),
            dtype=torch.long,
        )
        self.register_buffer("profiles", profiles, persistent=False)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(images), 1)

    def class_energies(self, features: torch.Tensor) -> torch.Tensor:
        """E_class(x, y) for all labels: [B, C]."""
        hidden = self.dropout(self.class_projection(features))
        return self.class_output(_gate(hidden[:, None, :], self.class_embedding[None])).squeeze(-1)

    def concept_energies(self, features: torch.Tensor) -> List[torch.Tensor]:
        """E_concept(x, c_g) for all states of each concept group: list of [B, K_g]."""
        hidden = self.dropout(self.concept_projection(features))
        return [
            output(_gate(hidden[:, None, :], embedding[None])).squeeze(-1)
            for output, embedding in zip(self.concept_outputs, self.concept_embeddings)
        ]

    def _global_energy(self, concept_vector: torch.Tensor, class_vector: torch.Tensor) -> torch.Tensor:
        """E_global(c, y) from concatenated concept vector [..., 14*d] and class vector [..., d]."""
        return self.global_output(_gate(self.global_projection(concept_vector), class_vector)).squeeze(-1)

    def concept_vector(self, concepts: torch.Tensor) -> torch.Tensor:
        """Maps discrete concepts [N, 14] to unnormalized concatenated embeddings [N, 14*d]."""
        return torch.cat(
            [embedding[concepts[:, group]] for group, embedding in enumerate(self.concept_embeddings)],
            dim=-1,
        )

    def global_class_energies(self, concepts: torch.Tensor) -> torch.Tensor:
        """E_global(c, y) for all classes given fixed concept profiles: [N, C]."""
        vector = self.concept_vector(concepts)
        return self._global_energy(vector[:, None, :], self.class_embedding[None])

    def relaxed_energy(
        self,
        features: torch.Tensor,
        concept_probabilities: List[torch.Tensor],
        class_probabilities: torch.Tensor,
        weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
        hard_global: bool = True,
    ) -> torch.Tensor:
        class_vector = class_probabilities @ self.class_embedding
        class_hidden = self.dropout(self.class_projection(features))
        concept_hidden = self.dropout(self.concept_projection(features))
        mixed = [
            prob @ embedding
            for prob, embedding in zip(concept_probabilities, self.concept_embeddings)
        ]
        e_class = self.class_output(_gate(class_hidden, class_vector)).squeeze(-1)
        e_concept = sum(
            output(_gate(concept_hidden, vector)).squeeze(-1)
            for output, vector in zip(self.concept_outputs, mixed)
        )
        if hard_global:
            global_vector = self.concept_vector(
                torch.stack([p.argmax(-1) for p in concept_probabilities], dim=-1)
            )
        else:
            global_vector = torch.cat(mixed, dim=-1)
        e_global = self._global_energy(global_vector, class_vector)
        w_class, w_concept, w_global = weights
        return w_class * e_class + w_concept * e_concept + w_global * e_global


def perturb_concepts(
    concepts: torch.Tensor,
    state_counts: Sequence[int],
    sample_fraction: float = 0.2,
    concept_fraction: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly flips concept states during training for E_global negative sampling."""
    perturbed = concepts.clone()
    if sample_fraction == 0 or concept_fraction == 0:
        return perturbed
    rows = torch.randperm(len(concepts), generator=generator)[:math.ceil(len(concepts) * sample_fraction)]
    groups = torch.randperm(len(state_counts), generator=generator)[:math.ceil(len(state_counts) * concept_fraction)]
    for group in groups.tolist():
        offsets = torch.randint(1, state_counts[group], (len(rows),), generator=generator)
        rows_device, offsets = rows.to(concepts.device), offsets.to(concepts.device)
        perturbed[rows_device, group] = (perturbed[rows_device, group] + offsets) % state_counts[group]
    return perturbed


def ecbm_training_loss(
    model: CategoricalECBM,
    images: torch.Tensor,
    concepts: torch.Tensor,
    labels: torch.Tensor,
    weights: Tuple[float, float, float] = (3.0, 1.0, 1.0),
    perturbation: Tuple[float, float] = (0.2, 0.2),
    class_weights: Optional[torch.Tensor] = None,
    concept_state_weights: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ECBM Training Loss from Xu et al. (ICLR 2024):
    L = lambda_xy * CE(-E_class, y) + lambda_xc * sum_g CE(-E_concept_g, c_g) + lambda_cy * CE(-E_global(c_perturbed), y)
    """
    features = model.extract_features(images)
    state_weights = concept_state_weights or [None] * len(model.state_counts)

    # 1. Image-Label loss
    class_loss = F.cross_entropy(-model.class_energies(features), labels, weight=class_weights)

    # 2. Image-Concept loss
    concept_loss = sum(
        F.cross_entropy(-energy, concepts[:, group], weight=state_weights[group])
        for group, energy in enumerate(model.concept_energies(features))
    )

    # 3. Global Concept-Label loss with perturbation
    global_concepts = (
        perturb_concepts(concepts, model.state_counts, *perturbation)
        if model.training
        else concepts
    )
    global_loss = F.cross_entropy(-model.global_class_energies(global_concepts), labels, weight=class_weights)

    w_class, w_concept, w_global = weights
    total = w_class * class_loss + w_concept * concept_loss + w_global * global_loss

    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite ECBM training loss encountered")

    return total, {
        "class": float(class_loss.detach()),
        "concept": float(concept_loss.detach()),
        "global": float(global_loss.detach()),
    }


@torch.no_grad()
def exact_inference(
    model: CategoricalECBM,
    features: torch.Tensor,
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    clamp: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Exact marginalization over all 16,384 discrete profiles:
        p(y, c | x) proportional to exp(-(w_class * E_class + w_concept * sum E_concept + w_global * E_global))
    Returns exact class probabilities p(y | x) and marginal concept probabilities p(c_g | x).
    When clamp is provided ({concept_idx: state_tensor}), computes exact conditional distribution under intervention!
    """
    w_class, w_concept, w_global = weights
    profiles = model.profiles.to(features.device)
    e_class = model.class_energies(features)
    e_concepts = model.concept_energies(features)
    e_global = model.global_class_energies(profiles)  # [P, C]

    concept_sum = sum(energy[:, profiles[:, group]] for group, energy in enumerate(e_concepts))  # [B, P]
    log_joint = -(w_class * e_class[:, None, :] + w_concept * concept_sum[:, :, None] + w_global * e_global[None])

    if clamp:
        allowed = torch.ones(len(features), len(profiles), dtype=torch.bool, device=features.device)
        for group, states in clamp.items():
            allowed &= profiles[None, :, group] == states.to(features.device)[:, None]
        log_joint = log_joint.masked_fill(~allowed[:, :, None], float("-inf"))

    log_joint = log_joint - torch.logsumexp(log_joint.flatten(1), dim=1)[:, None, None]
    class_probabilities = torch.logsumexp(log_joint, dim=1).exp()
    profile_probabilities = torch.logsumexp(log_joint, dim=2).exp()  # [B, P]

    concept_probabilities = [
        torch.zeros(len(features), count, device=features.device).index_add_(
            1, profiles[:, group], profile_probabilities
        )
        for group, count in enumerate(model.state_counts)
    ]
    return {
        "class_probabilities": class_probabilities,
        "concept_probabilities": concept_probabilities,
    }
