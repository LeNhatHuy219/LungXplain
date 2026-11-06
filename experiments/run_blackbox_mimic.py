#!/usr/bin/env python3
"""
Train, evaluate, or smoke-test the Black-box Classifier (EfficientNet-B0) on MIMIC-CXR
predicting In-Hospital Mortality (MIMIC-IV hospital_expire_flag).
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

from src.dataset import MimicCxrConceptDataset
from src.metrics import binary_metrics, select_validation_threshold
from src.models import BlackBoxClassifier


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


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


def run_pass(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, loss_denominator = 0.0, 0.0
    targets, probabilities, records = [], [], []

    with torch.enable_grad() if training else torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss encountered during training")

            if training:
                loss.backward()
                optimizer.step()

            batch_weights = criterion.weight[labels]
            denominator = float(batch_weights.sum().item())
            loss_sum += float(loss.item()) * denominator
            loss_denominator += denominator

            scores = logits.detach().softmax(dim=1)[:, 1].cpu().tolist()
            truth = labels.detach().cpu().tolist()
            targets.extend(truth)
            probabilities.extend(scores)

            meta = batch["meta"]
            for i, (target, score) in enumerate(zip(truth, scores)):
                records.append({
                    "dicom_id": meta["dicom_id"][i],
                    "study_id": meta["study_id"][i],
                    "subject_id": meta["subject_id"][i],
                    "target": target,
                    "probability_mortality": score,
                })

            if training and (step + 1) % 50 == 0:
                print(f"  train batch {step + 1}/{len(loader)} loss={loss_sum / loss_denominator:.4f}", flush=True)

    if not targets:
        raise ValueError("DataLoader produced zero samples")

    epoch_loss = loss_sum / loss_denominator if loss_denominator > 0 else 0.0
    return {
        "loss": epoch_loss,
        "targets": targets,
        "probabilities": probabilities,
        "records": records,
    }


def make_optimizer(model, args):
    return torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


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
        "architecture": "Black-box Classifier (EfficientNet-B0) -> Linear(1280,128) -> ReLU -> Dropout(0.2) -> Linear(128,2)",
        "pretrained_weights": "IMAGENET1K_V1" if not args.no_pretrained else None,
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

    criterion, weights = compute_class_weights(train_loader.dataset, device)
    model = BlackBoxClassifier(pretrained=not args.no_pretrained).to(device)
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = configuration(args, device, weights)
    write_json(out / "config.json", config)

    best_score, best_epoch = -1.0, None
    checkpoint_path = out / "best.pt"
    history_fields = [
        "epoch", "train_loss", "validation_loss", "validation_macro_f1_at_0_5",
        "validation_balanced_accuracy_at_0_5", "validation_roc_auc", "validation_pr_auc",
        "backbone_lr", "head_lr", "seconds"
    ]

    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            trained = run_pass(model, train_loader, criterion, device, optimizer=optimizer)
            validated = run_pass(model, val_loader, criterion, device)
            metrics = binary_metrics(validated["targets"], validated["probabilities"])

            row = {
                "epoch": epoch,
                "train_loss": trained["loss"],
                "validation_loss": validated["loss"],
                "validation_macro_f1_at_0_5": metrics["macro_f1"],
                "validation_balanced_accuracy_at_0_5": metrics["balanced_accuracy"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_pr_auc": metrics["pr_auc"],
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "seconds": time.monotonic() - start,
            }
            writer.writerow(row)
            handle.flush()

            if metrics["macro_f1"] > best_score + 1e-12:
                best_score, best_epoch = metrics["macro_f1"], epoch
                save_checkpoint(checkpoint_path, {
                    "format_version": 1,
                    "model_type": "blackbox",
                    "config": config,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "validation_macro_f1_at_0_5": best_score,
                    "threshold": None,
                })

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d}: train_loss={trained['loss']:.4f} "
                f"val_macro_f1={metrics['macro_f1']:.4f} val_bacc={metrics['balanced_accuracy']:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            scheduler.step()

    # Load best checkpoint and perform threshold calibration
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validated = run_pass(model, val_loader, criterion, device)
    threshold, sweep = select_validation_threshold(validated["targets"], validated["probabilities"])
    checkpoint["threshold"] = threshold
    save_checkpoint(checkpoint_path, checkpoint)

    write_csv(out / "validation_threshold_sweep.csv", sweep)
    save_predictions(out / "validation_predictions.csv", validated, threshold)

    result = {
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation_at_0_5": binary_metrics(validated["targets"], validated["probabilities"], 0.5),
        "validation_at_selected_threshold": binary_metrics(validated["targets"], validated["probabilities"], threshold),
        "checkpoint": str(checkpoint_path),
    }
    write_json(out / "validation_metrics.json", result)
    print(f"\n[BlackBox] Training finished. Best checkpoint: {checkpoint_path}", flush=True)
    print(f"[BlackBox] Best epoch: {best_epoch}, Validation threshold: {threshold:.2f}", flush=True)

    # Automatically evaluate on test set
    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    tested = run_pass(model, test_loader, criterion, device)
    test_result = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "threshold": threshold,
        "test_loss": tested["loss"],
        "test_at_0_5": binary_metrics(tested["targets"], tested["probabilities"], 0.5),
        "test_at_validation_threshold": binary_metrics(tested["targets"], tested["probabilities"], threshold),
    }
    write_json(out / "test_metrics.json", test_result)
    save_predictions(out / "test_predictions.csv", tested, threshold)
    print(f"\n[BlackBox Test Metrics at T={threshold:.2f}]:", flush=True)
    print(json.dumps(test_result["test_at_validation_threshold"], indent=2), flush=True)


def evaluate(args, manifest_path, data_root, out, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("threshold") is None:
        raise ValueError("Checkpoint has no validation-calibrated threshold; finish training first")

    args.image_size = checkpoint["config"]["image_size"]
    test_loader = make_loader(manifest_path, data_root, "test", args, device)
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    criterion, _ = compute_class_weights(train_loader.dataset, device)

    model = BlackBoxClassifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    tested = run_pass(model, test_loader, criterion, device)
    threshold = checkpoint["threshold"]

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "test_loss": tested["loss"],
        "test_at_0_5": binary_metrics(tested["targets"], tested["probabilities"], 0.5),
        "test_at_validation_threshold": binary_metrics(tested["targets"], tested["probabilities"], threshold),
    }
    write_json(out / "test_metrics.json", result)
    save_predictions(out / "test_predictions.csv", tested, threshold)
    print(f"\n[BlackBox Test Metrics at T={threshold:.2f}]:")
    print(json.dumps(result["test_at_validation_threshold"], indent=2), flush=True)


def smoke(args, manifest_path, data_root, out, device):
    train_loader = make_loader(manifest_path, data_root, "train", args, device)
    val_loader = make_loader(manifest_path, data_root, "valid", args, device)
    criterion, weights = compute_class_weights(train_loader.dataset, device)

    model = BlackBoxClassifier(pretrained=not args.no_pretrained).to(device)
    optimizer = make_optimizer(model, args)
    before = model.classifier[-1].weight.detach().cpu().clone()

    trained = run_pass(model, train_loader, criterion, device, optimizer=optimizer, max_batches=args.smoke_steps)
    validated = run_pass(model, val_loader, criterion, device, max_batches=1)

    update = (model.classifier[-1].weight.detach().cpu() - before).abs().sum().item()
    if not np.isfinite(update) or update <= 0:
        raise RuntimeError("Smoke test did not produce a parameter update")

    result = {
        "mode": "smoke",
        "device": str(device),
        "train_samples": len(trained["targets"]),
        "validation_samples": len(validated["targets"]),
        "train_loss": trained["loss"],
        "validation_loss": validated["loss"],
        "parameter_update_l1": update,
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{args.mode}_seed{args.seed}_e{args.epochs}_{stamp}"
    out = (args.output_dir or ROOT / "results/mimic_cxr/blackbox" / default_name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Executing BlackBox Runner | Mode: {args.mode} | Device: {device} | Seed: {args.seed}")
    if args.mode == "train":
        train(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "evaluate":
        evaluate(args, args.manifest_path, args.data_root, out, device)
    elif args.mode == "smoke":
        smoke(args, args.manifest_path, args.data_root, out, device)


if __name__ == "__main__":
    main()
