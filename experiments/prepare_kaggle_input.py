"""Package the validated v5 dataset as a private Kaggle input for the three notebooks.

Writes, into a new empty directory under data/:
  lungxplain_v5_data.tar   dataset files, resized pixel cache and ImageNet DenseNet-121 weights
  bundle_manifest_v5.json  SHA-256 of every member (the notebooks verify each file against it)
  dataset-metadata.json    metadata for `kaggle datasets create` (private by default)

The notebooks pin the SHA-256 of bundle_manifest_v5.json (EXPECTED_BUNDLE_SHA / EXPECTED_BUNDLE)
and of the v5 manifest (EXPECTED_MANIFEST_SHA / EXPECTED_MANIFEST). Replace the bundle pin with
the value printed here; the manifest pin matches only if the cohort was reproduced exactly.
No source JPEGs are included. Keep the Kaggle dataset private (MIMIC data use agreement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/mimic_cxr/multimodal_v5"
CACHE = ROOT / "data/model_cache/cxr_resized256_v5"
WEIGHTS = ROOT / "data/model_cache/torch/checkpoints/densenet121-a639ec97.pth"
DATA_FILES = ["manifest.csv", "manifest_temporal_sensitivity.csv", "feature_schema.json",
              "report.json", "validation_report.json", "image_audit.csv.gz"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ensure_weights() -> None:
    if WEIGHTS.is_file():
        return
    os.environ["TORCH_HOME"] = str(WEIGHTS.parents[1])
    from torchvision.models import DenseNet121_Weights, densenet121
    densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)


def main(args) -> None:
    output = args.output.resolve()
    output.relative_to(ROOT / "data")  # patient-level files stay under the ignored data/ directory
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a new empty output directory")
    validation = json.loads((DATASET / "validation_report.json").read_text())
    if validation.get("passed") is not True:
        raise SystemExit("Run experiments/validate_dataset_v5.py --write first; it must pass")
    manifest_sha = sha256(DATASET / "manifest.csv")
    index = json.loads((CACHE / "index.json").read_text())
    if index["configuration"]["manifest_sha256"] != manifest_sha or index["rows"] != 5000:
        raise SystemExit("The pixel cache does not belong to this manifest; rebuild it")
    ensure_weights()
    if not sha256(WEIGHTS).startswith("a639ec97"):
        raise SystemExit("Unexpected DenseNet-121 ImageNet weights")
    data = [DATASET / name for name in DATA_FILES] + [WEIGHTS] + sorted(CACHE.iterdir())
    members = {p.relative_to(ROOT).as_posix(): sha256(p) for p in data}
    aliases = {name: name + ".bin" for name in members if name.endswith(".gz")}
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output / "lungxplain_v5_data.tar", "w") as archive:
        for path in data:
            name = path.relative_to(ROOT).as_posix()
            archive.add(path, arcname=aliases.get(name, name), recursive=False)
    if any(sha256(ROOT / name) != value for name, value in members.items()):
        raise SystemExit("An input changed while the archive was being written")
    write_json(output / "bundle_manifest_v5.json", {
        "format": "lungxplain-private-v5", "dataset_version": "v5", "private_required": True,
        "dataset_dir": DATASET.relative_to(ROOT).as_posix(), "pixel_cache": CACHE.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha, "members": members, "aliases": aliases,
        "data_tar_sha256": sha256(output / "lungxplain_v5_data.tar"),
        "source_jpegs_included": False, "test_labels_used_for_training": False})
    write_json(output / "dataset-metadata.json", {
        "title": args.slug, "id": f"{args.owner}/{args.slug}", "licenses": [{"name": "other"}],
        "description": "Private credentialed MIMIC inputs for LungXplain v5. Do not make public."})
    print(json.dumps({"members": len(members), "manifest_sha256": manifest_sha,
                      "bundle_manifest_sha256 (paste into the notebooks)": sha256(output / "bundle_manifest_v5.json"),
                      "next": f"kaggle datasets create -p {output.relative_to(ROOT)}"}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Your Kaggle username")
    parser.add_argument("--slug", default="lungxplain-v5-private-input")
    parser.add_argument("--output", type=Path, default=ROOT / "data/kaggle_input")
    main(parser.parse_args())
