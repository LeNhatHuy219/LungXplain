#!/usr/bin/env python3
"""
Train, evaluate, or smoke-test Concept Bottleneck Models (CBM) on MIMIC-CXR:
- Soft CBM (continuous probabilities)
- Hard CBM (categorical one-hot bottleneck)
Standardized for 14 CheXpert concepts and In-Hospital Mortality prediction.
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

from src.concept_loss import build_concept_criteria, compute_concept_state_weights
from src.dataset import CHEXPERT_14_CONCEPTS, MimicCxrConceptDataset
from src.metrics import binary_metrics, concept_metrics, select_validation_threshold
from src.models import HardCBM, HardJointCBM, SoftJointCBM


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
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor), weights


def build_cbm_model(model_type: str, pretrained: bool = True):
    if model_type in ("soft", "soft_cbm"):
        return SoftJointCBM(concept_names=CHEXPERT_14_CONCEPTS, num_classes=2, pretrained=pretrained)
    elif model_type in ("hard", "hard_cbm"):
        return HardJointCBM(concept_names=CHEXPERT_14_CONCEPTS, num_classes=2, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown CBM model type: {model_type}. Expected 'soft' or 'hard'.")


def make_optimizer(model, args):
    backbone_params = list(model.features.parameters())
    head_params = list(model.concept_heads.parameters()) + list(model.diagnosis_head.parameters())
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


def _concept_loss(
    concept_logits: Dict[str, torch.Tensor],
    concept_indices: torch.Tensor,
    criteria: Dict[str, nn.Module],
) -> torch.Tensor:
    losses = [
        criteria[name](concept_logits[name], concept_indices[:, i])
        for i, name in enumerate(CHEXPERT_14_CONCEPTS)
    ]
    return torch.stack(losses).mean()


def run_pass(model, loader, criterion_y, criteria_c, lambda_c, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)

    loss_sum, loss_denominator = 0.0, 0.0
    task_loss_sum, concept_loss_sum = 0.0, 0.0
    targets_y, probs_y, records = [], [], []
    targets_c_list, probs_c_list = [], []

    with torch.enable_grad() if training else torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            c_gt = batch["concept_indices"].to(device)  # (B, 14) long

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits_y, concept_logits, concept_vector = model(images)

            loss_y = criterion_y(logits_y, labels)
            loss_c = _concept_loss(concept_logits, c_gt, criteria_c)
            total_loss = loss_y + lambda_c * loss_c

            if not torch.isfinite(total_loss):
                raise FloatingPointError("Nonfinite loss encountered during training")

            if training:
                total_loss.backward()
                optimizer.step()

            batch_weights = criterion_y.weight[labels]
            denominator = float(batch_weights.sum().item())
            loss_sum += float(total_loss.item()) * denominator
            task_loss_sum += float(loss_y.item()) * denominator
            concept_loss_sum += float(loss_c.item()) * denominator
            loss_denominator += denominator

            scores_y = logits_y.detach().softmax(dim=1)[:, 1].cpu().tolist()
            truth_y = labels.detach().cpu().tolist()
            targets_y.extend(truth_y)
            probs_y.extend(scores_y)

            # Probabilities of state 1 (present) for each concept:
            # concept_vector is [B, 28], odd indices 1, 3, ..., 27 correspond to 'present':
            c_probs_present = concept_vector[:, 1::2].detach().cpu().numpy()
            targets_c_list.append(c_gt.detach().cpu().numpy())
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

            if training and (step + 1) % 50 == 0:
                print(f"  train batch {step + 1}/{len(loader)} total_loss={loss_sum / loss_denominator:.4f} "
                      f"loss_y={task_loss_sum / loss_denominator:.4f} loss_c={concept_loss_sum / loss_denominator:.4f}", flush=True)

    if not targets_y:
        raise ValueError("DataLoader produced zero samples")

    norm = loss_denominator if loss_denominator > 0 else 1.0
    all_targets_c = np.concatenate(targets_c_list, axis=0)
    all_probs_c = np.concatenate(probs_c_list, axis=0)

    return {
        "total_loss": loss_sum / norm,
        "task_loss": task_loss_sum / norm,
        "concept_loss": concept_loss_sum / norm,
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
    model_name_map = {
        "soft": "Soft CBM",
        "soft_cbm": "Soft CBM",
        "hard": "Hard CBM",
        "hard_cbm": "Hard CBM",
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "descriptive_model_name": model_name_map.get(args.model_type, args.model_type),
        "backbone": "EfficientNet-B0",
        "num_concepts": 14,
        "target_name": "In-Hospital Mortality (hospital_expire_flag)",
        "checkpoint_selection": "maximum_validation_macro_f1_at_0.5; tie keeps earliest epoch",
        "threshold_selection": "maximum_validation_balanced_accuracy; 0.05..0.95 step 0.01; tie nearest 0.5",
        "class_weights": class_weights,
        "device_used": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "machine": platform.machine(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return config


def train(args, manifest_path, data_root, out, device):
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    val_loader = make_loader(manifest_path, data_root, "valid", args, device)

    criterion_y, weights = compute_class_weights(train_loader.dataset, device)
    state_weights = compute_concept_state_weights(train_loader.dataset.df, CHEXPERT_14_CONCEPTS)
    criteria_c = build_concept_criteria(state_weights, CHEXPERT_14_CONCEPTS, device)

    model = build_cbm_model(args.model_type, pretrained=not args.no_pretrained).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = configuration(args, device, weights)
    write_json(out / "config.json", config)

    best_score, best_epoch = -1.0, None
    checkpoint_path = out / "best.pt"
    history_fields = [
        "epoch", "train_loss", "train_task_loss", "train_concept_loss",
        "val_loss", "val_macro_f1_at_0_5", "val_balanced_acc_at_0_5", "val_roc_auc",
        "val_concept_macro_f1", "val_concept_accuracy", "val_exact_match_ratio",
        "backbone_lr", "head_lr", "seconds"
    ]

    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            trained = run_pass(
                model, train_loader, criterion_y, criteria_c, args.lambda_concept, device, optimizer=optimizer
            )
            validated = run_pass(
                model, val_loader, criterion_y, criteria_c, args.lambda_concept, device
            )

            metrics_y = binary_metrics(validated["targets_y"], validated["probabilities_y"])
            metrics_c = concept_metrics(
                validated["targets_c"], validated["probabilities_c"], CHEXPERT_14_CONCEPTS
            )

            row = {
                "epoch": epoch,
                "train_loss": trained["total_loss"],
                "train_task_loss": trained["task_loss"],
                "train_concept_loss": trained["concept_loss"],
                "val_loss": validated["total_loss"],
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
                    "model_type": args.model_type,
                    "config": config,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_macro_f1_at_0_5": best_score,
                    "threshold": None,
                })

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d}: train_loss={trained['total_loss']:.4f} "
                f"val_y_f1={metrics_y['macro_f1']:.4f} val_c_f1={metrics_c['mean_concept_macro_f1']:.4f} "
                f"val_exact_match={metrics_c['exact_match_ratio']:.4f} best_epoch={best_epoch}",
                flush=True,
            )
            scheduler.step()

    # Load best checkpoint and tune decision threshold
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validated = run_pass(model, val_loader, criterion_y, criteria_c, args.lambda_concept, device)
    threshold, sweep = select_validation_threshold(validated["targets_y"], validated["probabilities_y"])
    checkpoint["threshold"] = threshold
    save_checkpoint(checkpoint_path, checkpoint)

    val_metrics_y_0_5 = binary_metrics(validated["targets_y"], validated["probabilities_y"], 0.5)
    val_metrics_y_thresh = binary_metrics(validated["targets_y"], validated["probabilities_y"], threshold)
    val_metrics_c = concept_metrics(validated["targets_c"], validated["probabilities_c"], CHEXPERT_14_CONCEPTS)

    write_csv(out / "validation_threshold_sweep.csv", sweep)
    save_predictions(out / "validation_predictions.csv", validated, threshold)

    result = {
        "model_type": args.model_type,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_y_at_0_5": val_metrics_y_0_5,
        "validation_y_at_selected_threshold": val_metrics_y_thresh,
        "validation_concept_metrics": val_metrics_c,
        "checkpoint": str(checkpoint_path),
    }
    write_json(out / "validation_metrics.json", result)
    print(f"\n[{args.model_type}] Training finished. Best checkpoint: {checkpoint_path}", flush=True)
    print(f"[{args.model_type}] Best epoch: {best_epoch}, Threshold: {threshold:.2f}, Val Macro-F1: {best_score:.4f}", flush=True)

    # Automatically evaluate on test set
    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    tested = run_pass(model, test_loader, criterion_y, criteria_c, args.lambda_concept, device)
    test_metrics_y_0_5 = binary_metrics(tested["targets_y"], tested["probabilities_y"], 0.5)
    test_metrics_y_thresh = binary_metrics(tested["targets_y"], tested["probabilities_y"], threshold)
    test_metrics_c = concept_metrics(tested["targets_c"], tested["probabilities_c"], CHEXPERT_14_CONCEPTS)

    test_result = {
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "test_total_loss": tested["total_loss"],
        "test_task_loss": tested["task_loss"],
        "test_concept_loss": tested["concept_loss"],
        "test_y_at_0_5": test_metrics_y_0_5,
        "test_y_at_validation_threshold": test_metrics_y_thresh,
        "test_concept_metrics": test_metrics_c,
    }
    write_json(out / "test_metrics.json", test_result)
    save_predictions(out / "test_predictions.csv", tested, threshold)
    print(f"\n[{args.model_type} Test Mortality Metrics at T={threshold:.2f}]:", flush=True)
    print(json.dumps(test_metrics_y_thresh, indent=2), flush=True)
    print(f"[{args.model_type} Test Mean Concept Macro-F1: {test_metrics_c['mean_concept_macro_f1']:.4f}, Exact Match: {test_metrics_c['exact_match_ratio']:.4f}]", flush=True)


def evaluate(args, manifest_path, data_root, out, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("threshold") is None:
        raise ValueError("Checkpoint has no validation-calibrated threshold; finish training first")

    args.image_size = checkpoint["config"]["image_size"]
    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    criterion_y, _ = compute_class_weights(train_loader.dataset, device)
    state_weights = compute_concept_state_weights(train_loader.dataset.df, CHEXPERT_14_CONCEPTS)
    criteria_c = build_concept_criteria(state_weights, CHEXPERT_14_CONCEPTS, device)

    model_type = checkpoint.get("model_type", args.model_type)
    model = build_cbm_model(model_type, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    tested = run_pass(model, test_loader, criterion_y, criteria_c, args.lambda_concept, device)
    threshold = checkpoint["threshold"]

    test_metrics_y_0_5 = binary_metrics(tested["targets_y"], tested["probabilities_y"], 0.5)
    test_metrics_y_thresh = binary_metrics(tested["targets_y"], tested["probabilities_y"], threshold)
    test_metrics_c = concept_metrics(tested["targets_c"], tested["probabilities_c"], CHEXPERT_14_CONCEPTS)

    result = {
        "model_type": model_type,
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "test_total_loss": tested["total_loss"],
        "test_task_loss": tested["task_loss"],
        "test_concept_loss": tested["concept_loss"],
        "test_y_at_0_5": test_metrics_y_0_5,
        "test_y_at_validation_threshold": test_metrics_y_thresh,
        "test_concept_metrics": test_metrics_c,
    }
    write_json(out / "test_metrics.json", result)
    save_predictions(out / "test_predictions.csv", tested, threshold)

    print(f"\n[{model_type} Test Mortality Metrics at T={threshold:.2f}]:")
    print(json.dumps(test_metrics_y_thresh, indent=2), flush=True)
    print(f"\n[{model_type} Test Concept Metrics (Mean Macro-F1: {test_metrics_c['mean_concept_macro_f1']:.4f}, Exact Match: {test_metrics_c['exact_match_ratio']:.4f})]:", flush=True)


def smoke(args, manifest_path, data_root, out, device):
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    val_loader = make_loader(manifest_path, data_root, "valid", args, device)
    criterion_y, weights = compute_class_weights(train_loader.dataset, device)
    state_weights = compute_concept_state_weights(train_loader.dataset.df, CHEXPERT_14_CONCEPTS)
    criteria_c = build_concept_criteria(state_weights, CHEXPERT_14_CONCEPTS, device)

    model = build_cbm_model(args.model_type, pretrained=not args.no_pretrained).to(device)
    optimizer = make_optimizer(model, args)
    
    # Save parameter snapshots
    diag_head_last = list(model.diagnosis_head.parameters())[-1]
    first_c_head = list(model.concept_heads.values())[0]
    before_clf = diag_head_last.detach().cpu().clone()
    before_c = first_c_head.weight.detach().cpu().clone()

    trained = run_pass(
        model, train_loader, criterion_y, criteria_c, args.lambda_concept, device,
        optimizer=optimizer, max_batches=args.smoke_steps
    )
    validated = run_pass(
        model, val_loader, criterion_y, criteria_c, args.lambda_concept, device, max_batches=1
    )

    update_clf = (diag_head_last.detach().cpu() - before_clf).abs().sum().item()
    update_c = (first_c_head.weight.detach().cpu() - before_c).abs().sum().item()

    result = {
        "mode": "smoke",
        "model_type": args.model_type,
        "device": str(device),
        "train_samples": len(trained["targets_y"]),
        "train_loss": trained["total_loss"],
        "classifier_param_update_l1": update_clf,
        "concept_head_param_update_l1": update_c,
    }
    write_json(out / "smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train", "evaluate"], default="train")
    parser.add_argument("--model-type", choices=["soft", "hard", "soft_cbm", "hard_cbm"], default="soft")
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
    parser.add_argument("--lambda-concept", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{args.model_type}_seed{args.seed}_e{args.epochs}_{stamp}"
    out = (args.output_dir or ROOT / f"results/mimic_cxr/{args.model_type}" / default_name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Executing CBM Runner | Model: {args.model_type} | Mode: {args.mode} | Device: {device} | Seed: {args.seed}")
    if args.mode == "train":
        train(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "evaluate":
        evaluate(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "smoke":
        smoke(args, args.manifest_path, args.data_root, out, device)


if __name__ == "__main__":
    main()
