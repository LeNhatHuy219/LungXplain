"""Full image-level DenseNet121 fine-tuning with resumable, fold-local selection.

Frozen-feature baseline code stays unchanged. Pixel caches contain deterministic
resized images, never learned representations; every training batch backpropagates
through the entire CNN. All patient-linked artifacts remain local under data/results.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import InterpolationMode

from .data import ROOT, sha256, write_json


@dataclass(frozen=True)
class FineTuneConfig:
    max_epochs: int = 60
    min_epochs: int = 10
    patience: int = 10
    batch_size: int = 16
    backbone_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-5
    image_size: int = 224
    resize_size: int = 256
    label_policy: str = "u_zeros"
    num_workers: int = 0
    grad_clip: float = 5.0
    seed: int = 42


class FineTunedConceptNet(nn.Module):
    def __init__(self, concepts=14, pretrained=True):
        super().__init__()
        torch.hub.set_dir(str(ROOT / "data/model_cache/torch"))
        original = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = original.features
        self.classifier = nn.Linear(original.classifier.in_features, concepts)
        self.requires_grad_(True)

    def forward(self, images):
        spatial = torch.relu(self.features(images))
        pooled = nn.functional.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1)
        return self.classifier(pooled)


def transforms_for(config, training=False, include_resize=False):
    operations = []
    if include_resize:
        operations.append(transforms.Resize(config.resize_size, interpolation=InterpolationMode.BILINEAR))
    if training:
        operations.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=0))
    operations.extend([transforms.CenterCrop(config.image_size), transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transforms.Compose(operations)


def label_mask(cohort, policy):
    if policy == "u_zeros":
        return np.ones_like(cohort.concepts(), dtype=bool)
    if policy == "u_ignore":
        uncertain = cohort.frame[["ignore_" + name for name in cohort.concept_names]].to_numpy(dtype=np.int8)
        return uncertain == 0
    if policy == "known":
        return cohort.known()
    raise ValueError("Unknown concept supervision policy")


def prepare_pixel_cache(cohort, config, directory):
    directory = Path(directory)
    metadata_path = directory / "index.json"
    expected = {"manifest_sha256": cohort.manifest_hash, "resize_size": config.resize_size,
                "interpolation": "PIL_RGB_BILINEAR", "learned_features": False,
                "source_image_audit_sha256": sha256(cohort.dataset_dir / "image_audit.csv.gz")}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["configuration"] != expected:
            raise ValueError("Pixel cache belongs to different inputs; choose a new cache directory")
        for record in metadata["images"]:
            if sha256(directory / record["file"]) != record["sha256"]:
                raise ValueError("Resized pixel cache was modified")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    # The directory is specific to this manifest/resize; a partial build can be
    # regenerated atomically without touching original medical JPEGs.
    image_audit = __import__("pandas").read_csv(cohort.dataset_dir / "image_audit.csv.gz", dtype=str, keep_default_na=False)
    references = dict(zip(image_audit["dicom_id"], image_audit["sha256"]))
    resize = transforms.Resize(config.resize_size, interpolation=InterpolationMode.BILINEAR)

    def cache_one(item):
        index, row = item
        source = ROOT / row["image_path"]
        if sha256(source) != references[row["dicom_id"]]:
            raise ValueError("Source JPEG changed since dataset validation")
        target = directory / f"{index:05d}.png"
        with Image.open(source) as image:
            resized = resize(image.convert("RGB"))
            temporary = target.with_suffix(".png.partial")
            resized.save(temporary, format="PNG", compress_level=1)
            temporary.replace(target)
        return {"file": target.name, "sha256": sha256(target)}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(cache_one, enumerate(cohort.frame.to_dict("records"))):
            records.append(record)
            if len(records) % 500 == 0:
                print(json.dumps({"pixel_cache_images": len(records), "total": len(cohort.frame)}), flush=True)
    write_json(metadata_path, {"configuration": expected, "images": records, "rows": len(records)})
    return directory


class ImageConceptRows(Dataset):
    def __init__(self, cache, indices, labels, mask, config, training=False):
        self.cache = Path(cache)
        self.indices = np.asarray(indices, dtype=int)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=np.float32)
        self.transform = transforms_for(config, training=training)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        with Image.open(self.cache / f"{index:05d}.png") as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.labels[index]), torch.from_numpy(self.mask[index]), int(index)


def masked_bce(logits, labels, mask):
    values = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (values * mask).sum() / mask.sum().clamp_min(1)


def concept_auc(labels, probabilities, mask):
    scores = []
    for column in range(labels.shape[1]):
        y = labels[mask[:, column].astype(bool), column]
        p = probabilities[mask[:, column].astype(bool), column]
        if np.sum(y == 0) >= 2 and np.sum(y == 1) >= 2:
            scores.append(float(roc_auc_score(y, p)))
    return float(np.mean(scores)) if scores else None, len(scores)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(cpu_tree(value), temporary)
    temporary.replace(path)


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.features.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state(device):
    result = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def emit(directory, **message):
    message = {"stage": directory.name, **message}
    print(json.dumps(message), flush=True)
    write_json(directory / "progress.json", message)


def evaluate_cnn(model, loader, device):
    model.eval()
    probabilities, targets, masks, indices = [], [], [], []
    loss_sum = count = 0.0
    with torch.inference_mode():
        for images, labels, mask, row in loader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            logits = model(images)
            weight = float(mask.sum().item())
            loss_sum += float(masked_bce(logits, labels, mask).item()) * weight
            count += weight
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())
            masks.append(mask.cpu().numpy())
            indices.append(row.numpy())
    predicted = np.concatenate(probabilities)
    labels, mask = np.concatenate(targets), np.concatenate(masks)
    auc, eligible = concept_auc(labels, predicted, mask)
    return {"loss": loss_sum / max(count, 1), "macro_auroc": auc, "auroc_concepts": eligible,
            "probabilities": predicted, "indices": np.concatenate(indices)}


def fit_cnn_stage(cohort, cache, fitting, selection, config, directory, device, *, fixed_epochs=None):
    """Every gradient and checkpoint-selection label belongs to outer fitting.

    With selection=None, refit all 4000 train rows for a fixed epoch budget chosen
    from the outer folds' inner validation. Global validation/test never select CNN.
    """
    import dataclasses
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fitting = np.asarray(fitting, dtype=int)
    selection = np.asarray(selection, dtype=int) if selection is not None else np.array([], dtype=int)
    allowed = set(cohort.indices("train"))
    if not set(fitting) <= allowed or not set(selection) <= allowed or set(fitting) & set(selection):
        raise ValueError("CNN fitting/selection must be disjoint training-only patients")
    if not len(fitting) or (fixed_epochs is None and not len(selection)):
        raise ValueError("An inner validation set or a fixed final epoch budget is required")
    budget = int(fixed_epochs or config.max_epochs)
    binding = {"configuration": dataclasses.asdict(config), "manifest_sha256": cohort.manifest_hash,
               "fitting_indices": fitting.tolist(), "selection_indices": selection.tolist(),
               "epoch_budget": budget, "selection_metric": "inner_concept_macro_auroc" if len(selection) else "fixed_train_only_refit",
               "trainer_sha256": sha256(Path(__file__))}
    completed_path = directory / "completed.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        if completed["binding"] != binding or sha256(directory / "best.pt") != completed["best_checkpoint_sha256"]:
            raise ValueError("Completed CNN stage does not match this run")
        return completed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    labels, mask = cohort.concepts(), label_mask(cohort, config.label_policy)
    model = FineTunedConceptNet(len(cohort.concept_names), pretrained=True)
    with torch.no_grad():
        supervised = mask[fitting].sum(0)
        positive = (labels[fitting] * mask[fitting]).sum(0)
        prior = np.clip((positive + 0.5) / (supervised + 1), 1e-4, 1 - 1e-4)
        nn.init.normal_(model.classifier.weight, std=0.01)
        model.classifier.bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
    initial_hash = backbone_hash(model)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": config.backbone_lr},
        {"params": model.classifier.parameters(), "lr": config.head_lr}], weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    generator = torch.Generator()
    train_loader = DataLoader(ImageConceptRows(cache, fitting, labels, mask, config, training=True),
                              batch_size=config.batch_size, shuffle=True, generator=generator,
                              num_workers=config.num_workers, persistent_workers=config.num_workers > 0)
    val_loader = DataLoader(ImageConceptRows(cache, selection, labels, mask, config),
                            batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers,
                            persistent_workers=config.num_workers > 0) if len(selection) else None
    start_epoch, best_score, best_epoch, stale = 1, -float("inf"), 0, 0
    update_steps = 0
    last_path = directory / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["binding"] != binding:
            raise ValueError("Checkpoint/configuration mismatch; do not resume with changed data or code")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_score, best_epoch, stale = state["best_score"], state["best_epoch"], state["stale"]
        update_steps, initial_hash = state["update_steps"], state["initial_backbone_sha256"]
        restore_rng(state["rng_state"], device)
        history_path = directory / "history.jsonl"
        logged = {json.loads(line)["epoch"] for line in history_path.read_text().splitlines() if line} if history_path.exists() else set()
        if state["epoch"] not in logged:
            with history_path.open("a") as stream:
                stream.write(json.dumps(state["summary"]) + "\n")
        emit(directory, event="resumed", next_epoch=start_epoch, epoch_budget=budget)
    emit(directory, event="cnn_finetuning_started", fitting_rows=len(fitting), inner_validation_rows=len(selection),
         epoch_budget=budget, trainable_backbone_parameters=sum(p.numel() for p in model.features.parameters() if p.requires_grad), device=str(device))
    last_epoch = start_epoch - 1
    already_stopped = bool(len(selection) and start_epoch - 1 >= config.min_epochs and stale >= config.patience)
    epochs_to_run = [] if already_stopped else range(start_epoch, budget + 1)
    for epoch in epochs_to_run:
        epoch_started = time.monotonic()
        generator.manual_seed(config.seed * 1000 + epoch)
        model.train()
        loss_sum = label_count = 0.0
        examples = 0
        backbone_grad_nonzero = False
        for step, (images, target, observed, _) in enumerate(train_loader, 1):
            images, target, observed = images.to(device), target.to(device), observed.to(device)
            if observed.sum().item() == 0:
                examples += len(images)
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = masked_bce(logits, target, observed)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite fine-tuning loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not backbone_grad_nonzero:
                grad = model.features.conv0.weight.grad
                backbone_grad_nonzero = grad is not None and bool(torch.any(grad != 0).item())
                if not backbone_grad_nonzero:
                    raise RuntimeError("CNN backbone received no gradient; refusing a frozen-backbone run")
            optimizer.step()
            update_steps += 1
            weight = float(observed.sum().item())
            loss_sum += float(loss.detach().item()) * weight
            label_count += weight
            examples += len(images)
            if step % 20 == 0 or step == len(train_loader):
                emit(directory, event="training_batch", epoch=epoch, epoch_budget=budget, step=step,
                     steps=len(train_loader), training_loss=loss_sum / max(label_count, 1),
                     elapsed_seconds=round(time.monotonic() - epoch_started, 1))
        if examples != len(fitting):
            raise RuntimeError("Not every fitting patient was seen in this epoch")
        if not backbone_grad_nonzero:
            raise RuntimeError("No supervised backbone update occurred in this epoch")
        validation = evaluate_cnn(model, val_loader, device) if val_loader is not None else None
        score = (validation["macro_auroc"] if validation["macro_auroc"] is not None else -validation["loss"]) if validation else float(epoch)
        improved = score > best_score + 1e-5
        if improved:
            best_score, best_epoch, stale = float(score), epoch, 0
        else:
            stale += 1
        if validation is not None:
            scheduler.step(score)
        current_hash = backbone_hash(model)
        if current_hash == initial_hash:
            raise RuntimeError("CNN parameters did not change after fine-tuning")
        summary = {"epoch": epoch, "epoch_budget": budget, "training_loss": loss_sum / max(label_count, 1),
                   "inner_validation_loss": validation["loss"] if validation else None,
                   "inner_concept_macro_auroc": validation["macro_auroc"] if validation else None,
                   "inner_auroc_concepts": validation["auroc_concepts"] if validation else None,
                   "best_epoch": best_epoch, "best_score": best_score, "stale_epochs": stale,
                   "backbone_grad_nonzero": backbone_grad_nonzero, "backbone_parameters_changed": True,
                   "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
                   "update_steps": update_steps, "epoch_seconds": round(time.monotonic() - epoch_started, 3)}
        state = {"binding": binding, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(), "rng_state": rng_state(device),
                 "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                 "update_steps": update_steps, "initial_backbone_sha256": initial_hash,
                 "backbone_sha256": current_hash, "summary": summary}
        if improved:
            atomic_checkpoint(directory / "best.pt", {"binding": binding, "model_state_dict": model.state_dict(),
                              "epoch": epoch, "initial_backbone_sha256": initial_hash, "backbone_sha256": current_hash,
                              "update_steps": update_steps, "summary": summary})
        # Publish best first. A crash cannot leave a last checkpoint that refers
        # to a new best epoch whose weights have not reached disk yet.
        atomic_checkpoint(last_path, state)
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        emit(directory, event="epoch_complete", **summary)
        last_epoch = epoch
        if len(selection) and epoch >= config.min_epochs and stale >= config.patience:
            break
    result = {"binding": binding, "epochs_completed": last_epoch, "best_epoch": best_epoch,
              "best_score": best_score, "update_steps": update_steps,
              "best_checkpoint_sha256": sha256(directory / "best.pt"), "backbone_was_finetuned": True,
              "stopped_early": last_epoch < budget, "global_validation_used_for_checkpoint_selection": False}
    write_json(completed_path, result)
    emit(directory, event="cnn_stage_complete", epochs_completed=last_epoch, best_epoch=best_epoch,
         stopped_early=result["stopped_early"])
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def predict_cnn(checkpoint, cache, indices, cohort, config, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["initial_backbone_sha256"] == state["backbone_sha256"]:
        raise ValueError("Checkpoint has not fine-tuned the backbone")
    model = FineTunedConceptNet(len(cohort.concept_names), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    loader = DataLoader(ImageConceptRows(cache, indices, cohort.concepts(), label_mask(cohort, config.label_policy), config),
                        batch_size=config.batch_size * 2, shuffle=False, num_workers=config.num_workers)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, _, _, _ in loader:
            predictions.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(predictions)
