#!/usr/bin/env python3
"""Download the selected MIMIC-CXR-JPG images and verify each against the official SHA256SUMS.

Credentials come from environment variables (never hard-code them, never share them):
    export PHYSIONET_USER=...      # YOUR credentialed PhysioNet account
    export PHYSIONET_PASSWORD=...
    python experiments/download_images.py

Reads data/mimic_cxr/download_list.txt, writes into data/mimic_cxr/dataset/files/...
Safe to re-run: files that already exist and match their SHA256 are skipped.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mimic_cxr"
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
DEST = DATA_DIR / "dataset"


def load_checksums() -> dict[str, str]:
    sums = {}
    with open(DATA_DIR / "mimic-cxr-jpg-2.1.0-SHA256SUMS.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                sums[parts[1].lstrip("*")] = parts[0]
    return sums


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
    if not user or not password:
        sys.exit("Set PHYSIONET_USER and PHYSIONET_PASSWORD (your own credentialed account).")
    wanted = [l.strip() for l in (DATA_DIR / "download_list.txt").read_text().splitlines() if l.strip()]
    sums = load_checksums()
    session = requests.Session()
    session.auth = (user, password)

    ok = skipped = failed = 0
    for i, rel in enumerate(wanted, 1):
        target = DEST / rel
        expected = sums.get(rel)
        if expected is None:
            print(f"[no checksum] {rel}", file=sys.stderr)
            failed += 1
            continue
        if target.is_file() and sha256(target) == expected:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        try:
            with session.get(BASE_URL + rel, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if sha256(tmp) != expected:
                raise RuntimeError("SHA256 mismatch")
            tmp.replace(target)
            ok += 1
        except Exception as exc:  # keep going; re-run to retry
            tmp.unlink(missing_ok=True)
            print(f"[failed] {rel}: {exc}", file=sys.stderr)
            failed += 1
        if i % 100 == 0:
            print(f"{i}/{len(wanted)} ok={ok} skipped={skipped} failed={failed}", flush=True)
        time.sleep(0.05)
    print(f"done: ok={ok} skipped={skipped} failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
