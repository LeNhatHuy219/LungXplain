#!/usr/bin/env python3
"""
Train, evaluate, or smoke-test the Energy-Based Concept Bottleneck Model (ECBM) on MIMIC-CXR.
Follows the official Xu et al. (ICLR 2024) / dermaXAI formulation:
- Backbone: EfficientNet-B0
- Three energies: E_class(x, y), E_concept(x, c), E_global(c, y)
- Exact enumeration inference over all 2^14 = 16,384 discrete profiles.
- Standardized for 14 CheXpert concepts and In-Hospital Mortality prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from src.concept_loss import compute_concept_state_weights
from src.dataset import CHEXPERT_14_CONCEPTS, MimicCxrConceptDataset
from src.ecbm import CategoricalECBM, ecbm_training_loss, exact_inference
from src.metrics import binary_metrics, concept_metrics, select_validation_threshold


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def choose_device(requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process; use --device cpu or auto")
    return torch.device(requested)


def make_loader(manifest_path, data_root, split, args, device):
    dataset = MimicCxrConceptDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        image_size=args.image_size,
        training=split == "train",
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def compute_class_weights(dataset: MimicCxrConceptDataset, device: torch.device):
    labels = dataset.df["diagnosis_binary"].values
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = len(labels)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    weights = [float(w_neg), float(w_pos)]
    return torch.tensor(weights, dtype=torch.float32, device=device), weights


def compute_state_weights_tensors(dataset: MimicCxrConceptDataset, device: torch.device):
    weights_dict = compute_concept_state_weights(dataset.df, CHEXPERT_14_CONCEPTS)
    return [
        torch.tensor(weights_dict[name], dtype=torch.float32, device=device)
        for name in CHEXPERT_14_CONCEPTS
    ]


def make_optimizer(model: CategoricalECBM, args):
    backbone_params = list(model.features.parameters())
    energy_params = (
        [model.class_embedding]
        + list(model.concept_embeddings)
        + list(model.class_projection.parameters())
        + list(model.class_output.parameters())
        + list(model.concept_projection.parameters())
        + list(model.concept_outputs.parameters())
        + list(model.global_projection.parameters())
        + list(model.global_output.parameters())
    )
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": energy_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


def run_train_epoch(model, loader, optimizer, class_weights, concept_weights, weights, device, max_batches=None):
    model.train()
    loss_sum, loss_denom = 0.0, 0.0
    comp_sums = {"class": 0.0, "concept": 0.0, "global": 0.0}

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concepts = batch["concept_indices"].to(device)  # (B, 14)

        optimizer.zero_grad(set_to_none=True)
        total_loss, loss_components = ecbm_training_loss(
            model=model,
            images=images,
            concepts=concepts,
            labels=labels,
            weights=weights,
            class_weights=class_weights,
            concept_state_weights=concept_weights,
        )
        total_loss.backward()
        optimizer.step()

        b_weight = float(class_weights[labels].sum().item())
        loss_sum += float(total_loss.item()) * b_weight
        loss_denom += b_weight
        for k in comp_sums:
            comp_sums[k] += loss_components[k] * b_weight

        if (step + 1) % 50 == 0:
            print(f"  train batch {step + 1}/{len(loader)} total_loss={loss_sum / loss_denom:.4f}", flush=True)

    norm = loss_denom if loss_denom > 0 else 1.0
    return {
        "loss": loss_sum / norm,
        "class_loss": comp_sums["class"] / norm,
        "concept_loss": comp_sums["concept"] / norm,
        "global_loss": comp_sums["global"] / norm,
    }


@torch.no_grad()
def run_eval_epoch(model, loader, device, max_batches=None):
    model.eval()
    targets_y, probs_y, records = [], [], []
    targets_c_list, probs_c_list = [], []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        concepts = batch["concept_indices"].to(device)

        features = model.extract_features(images)
        exact_out = exact_inference(model, features)

        scores_y = exact_out["class_probabilities"][:, 1].cpu().tolist()
        truth_y = labels.cpu().tolist()
        targets_y.extend(truth_y)
        probs_y.extend(scores_y)

        # Marginal concept probabilities for present state (state 1):
        # concept_probabilities is list of 14 tensors each [B, 2]
        c_probs_present = torch.stack(
            [cp[:, 1] for cp in exact_out["concept_probabilities"]], dim=1
        ).cpu().numpy()

        targets_c_list.append(concepts.cpu().numpy())
        probs_c_list.append(c_probs_present)

        meta = batch["meta"]
        for i, (target, score) in enumerate(zip(truth_y, scores_y)):
            records.append({
                "dicom_id": meta["dicom_id"][i],
                "study_id": meta["study_id"][i],
                "subject_id": meta["subject_id"][i],
                "target": target,
                "probability_mortality": score,
            })

    all_targets_c = np.concatenate(targets_c_list, axis=0)
    all_probs_c = np.concatenate(probs_c_list, axis=0)

    return {
        "targets_y": targets_y,
        "probabilities_y": probs_y,
        "targets_c": all_targets_c,
        "probabilities_c": all_probs_c,
        "records": records,
    }


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_predictions(path, result, threshold):
    rows = [
        dict(row, prediction=int(row["probability_mortality"] >= threshold))
        for row in result["records"]
    ]
    write_csv(path, rows)


def configuration(args, device, class_weights=None):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "descriptive_model_name": "Energy-Based Concept Bottleneck Model (ECBM)",
        "backbone": "EfficientNet-B0",
        "num_concepts": 14,
        "inference_method": "Exact Enumeration over 16,384 Profiles",
        "target_name": "In-Hospital Mortality (hospital_expire_flag)",
        "checkpoint_selection": "maximum_validation_macro_f1_at_0.5; tie keeps earliest epoch",
        "threshold_selection": "maximum_validation_balanced_accuracy; 0.05..0.95 step 0.01; tie nearest 0.5",
        "class_weights": class_weights,
        "device_used": str(device),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return config


def train(args, manifest_path, data_root, out, device):
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    val_loader = make_loader(manifest_path, data_root, "valid", args, device)

    class_weight_tensor, weights_list = compute_class_weights(train_loader.dataset, device)
    concept_weight_tensors = compute_state_weights_tensors(train_loader.dataset, device)

    model = CategoricalECBM(
        concept_names=CHEXPERT_14_CONCEPTS,
        num_classes=2,
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = configuration(args, device, weights_list)
    write_json(out / "config.json", config)

    loss_weights = (args.w_class, args.w_concept, args.w_global)
    best_score, best_epoch = -1.0, None
    checkpoint_path = out / "best.pt"

    history_fields = [
        "epoch", "train_total_loss", "train_class_loss", "train_concept_loss", "train_global_loss",
        "val_macro_f1_at_0_5", "val_balanced_acc_at_0_5", "val_roc_auc",
        "val_concept_macro_f1", "val_concept_accuracy", "val_exact_match_ratio",
        "backbone_lr", "head_lr", "seconds"
    ]

    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            trained = run_train_epoch(
                model, train_loader, optimizer, class_weight_tensor, concept_weight_tensors,
                loss_weights, device
            )
            validated = run_eval_epoch(model, val_loader, device)

            metrics_y = binary_metrics(validated["targets_y"], validated["probabilities_y"])
            metrics_c = concept_metrics(
                validated["targets_c"], validated["probabilities_c"], CHEXPERT_14_CONCEPTS
            )

            row = {
                "epoch": epoch,
                "train_total_loss": trained["loss"],
                "train_class_loss": trained["class_loss"],
                "train_concept_loss": trained["concept_loss"],
                "train_global_loss": trained["global_loss"],
                "val_macro_f1_at_0_5": metrics_y["macro_f1"],
                "val_balanced_acc_at_0_5": metrics_y["balanced_accuracy"],
                "val_roc_auc": metrics_y["roc_auc"],
                "val_concept_macro_f1": metrics_c["mean_concept_macro_f1"],
                "val_concept_accuracy": metrics_c["mean_concept_accuracy"],
                "val_exact_match_ratio": metrics_c["exact_match_ratio"],
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "seconds": time.monotonic() - start,
            }
            writer.writerow(row)
            handle.flush()

            if metrics_y["macro_f1"] > best_score + 1e-12:
                best_score, best_epoch = metrics_y["macro_f1"], epoch
                save_checkpoint(checkpoint_path, {
                    "format_version": 1,
                    "model_type": "ecbm",
                    "config": config,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_macro_f1_at_0_5": best_score,
                    "threshold": None,
                })

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d}: train_loss={trained['loss']:.4f} "
                f"val_y_f1={metrics_y['macro_f1']:.4f} val_c_f1={metrics_c['mean_concept_macro_f1']:.4f} "
                f"val_exact_match={metrics_c['exact_match_ratio']:.4f} best_epoch={best_epoch}",
                flush=True,
            )
            scheduler.step()

    # Load best checkpoint and tune decision threshold
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validated = run_eval_epoch(model, val_loader, device)
    threshold, sweep = select_validation_threshold(validated["targets_y"], validated["probabilities_y"])
    checkpoint["threshold"] = threshold
    save_checkpoint(checkpoint_path, checkpoint)

    val_metrics_y_0_5 = binary_metrics(validated["targets_y"], validated["probabilities_y"], 0.5)
    val_metrics_y_thresh = binary_metrics(validated["targets_y"], validated["probabilities_y"], threshold)
    val_metrics_c = concept_metrics(validated["targets_c"], validated["probabilities_c"], CHEXPERT_14_CONCEPTS)

    write_csv(out / "validation_threshold_sweep.csv", sweep)
    save_predictions(out / "validation_predictions.csv", validated, threshold)

    result = {
        "model_type": "ecbm",
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_y_at_0_5": val_metrics_y_0_5,
        "validation_y_at_selected_threshold": val_metrics_y_thresh,
        "validation_concept_metrics": val_metrics_c,
        "checkpoint": str(checkpoint_path),
    }
    write_json(out / "validation_metrics.json", result)
    print(f"\n[ECBM] Training finished. Best checkpoint: {checkpoint_path}", flush=True)
    print(f"[ECBM] Best epoch: {best_epoch}, Threshold: {threshold:.2f}, Val Macro-F1: {best_score:.4f}", flush=True)

    # Automatically evaluate on test set
    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    tested = run_eval_epoch(model, test_loader, device)
    test_metrics_y_0_5 = binary_metrics(tested["targets_y"], tested["probabilities_y"], 0.5)
    test_metrics_y_thresh = binary_metrics(tested["targets_y"], tested["probabilities_y"], threshold)
    test_metrics_c = concept_metrics(tested["targets_c"], tested["probabilities_c"], CHEXPERT_14_CONCEPTS)

    test_result = {
        "model_type": "ecbm",
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "test_loss": tested["loss"],
        "test_y_at_0_5": test_metrics_y_0_5,
        "test_y_at_validation_threshold": test_metrics_y_thresh,
        "test_concept_metrics": test_metrics_c,
    }
    write_json(out / "test_metrics.json", test_result)
    save_predictions(out / "test_predictions.csv", tested, threshold)
    print(f"\n[ECBM Test Mortality Metrics at T={threshold:.2f}]:", flush=True)
    print(json.dumps(test_metrics_y_thresh, indent=2), flush=True)
    print(f"[ECBM Test Mean Concept Macro-F1: {test_metrics_c['mean_concept_macro_f1']:.4f}, Exact Match: {test_metrics_c['exact_match_ratio']:.4f}]", flush=True)


def evaluate(args, manifest_path, data_root, out, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    args.image_size = checkpoint["config"]["image_size"]

    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    model = CategoricalECBM(
        concept_names=CHEXPERT_14_CONCEPTS,
        num_classes=2,
        pretrained=False,
        hidden_dim=checkpoint["config"].get("hidden_dim", 32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    tested = run_eval_epoch(model, test_loader, device)
    threshold = checkpoint["threshold"]

    test_metrics_y_0_5 = binary_metrics(tested["targets_y"], tested["probabilities_y"], 0.5)
    test_metrics_y_thresh = binary_metrics(tested["targets_y"], tested["probabilities_y"], threshold)
    test_metrics_c = concept_metrics(tested["targets_c"], tested["probabilities_c"], CHEXPERT_14_CONCEPTS)

    result = {
        "model_type": "ecbm",
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "test_y_at_0_5": test_metrics_y_0_5,
        "test_y_at_validation_threshold": test_metrics_y_thresh,
        "test_concept_metrics": test_metrics_c,
    }
    write_json(out / "test_metrics.json", result)
    save_predictions(out / "test_predictions.csv", tested, threshold)

    print(f"\n[ECBM Test Mortality Metrics at T={threshold:.2f}]:")
    print(json.dumps(test_metrics_y_thresh, indent=2), flush=True)
    print(f"\n[ECBM Test Concept Metrics (Mean Macro-F1: {test_metrics_c['mean_concept_macro_f1']:.4f}, Exact Match: {test_metrics_c['exact_match_ratio']:.4f})]:", flush=True)


def smoke(args, manifest_path, data_root, out, device):
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    val_loader = make_loader(manifest_path, data_root, "valid", args, device)

    class_weight_tensor, weights_list = compute_class_weights(train_loader.dataset, device)
    concept_weight_tensors = compute_state_weights_tensors(train_loader.dataset, device)

    model = CategoricalECBM(
        concept_names=CHEXPERT_14_CONCEPTS,
        num_classes=2,
        pretrained=not args.no_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = make_optimizer(model, args)
    before_class_emb = model.class_embedding.detach().cpu().clone()

    loss_weights = (args.w_class, args.w_concept, args.w_global)
    trained = run_train_epoch(
        model, train_loader, optimizer, class_weight_tensor, concept_weight_tensors,
        loss_weights, device, max_batches=args.smoke_steps
    )
    validated = run_eval_epoch(model, val_loader, device, max_batches=1)

    update = (model.class_embedding.detach().cpu() - before_class_emb).abs().sum().item()
    result = {
        "mode": "smoke",
        "model_type": "ecbm",
        "device": str(device),
        "train_samples": args.smoke_steps * args.batch_size,
        "train_loss": trained["loss"],
        "class_emb_param_update_l1": update,
    }
    write_json(out / "smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "evaluate"], default="train")
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "data/mimic_cxr/manifest_mortality.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mimic_cxr")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--w-class", type=float, default=3.0)
    parser.add_argument("--w-concept", type=float, default=1.0)
    parser.add_argument("--w-global", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"ecbm_seed{args.seed}_e{args.epochs}_{stamp}"
    out = (args.output_dir or ROOT / "results/mimic_cxr/ecbm" / default_name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Executing ECBM Runner | Mode: {args.mode} | Device: {device} | Seed: {args.seed}")
    if args.mode == "train":
        train(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "evaluate":
        evaluate(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "smoke":
        smoke(args, args.manifest_path, args.data_root, out, device)


if __name__ == "__main__":
    main()
