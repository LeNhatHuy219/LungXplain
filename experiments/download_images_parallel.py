"""Parallel download of a cohort's missing MIMIC-CXR-JPG images with the user's PhysioNet access.

Same guarantees as download_5000_replacements.py: one web login (password prompt
hidden, never saved; PHYSIONET_USER/PHYSIONET_PASSWORD accepted), every file is
checked against the published SHA-256 and decoded as JPEG before it is kept, and
files already verified are skipped, so the command can be re-run after an
interruption. Several images are fetched at once (default 8 connections; keep it
modest to respect PhysioNet). Only aggregate progress is printed.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multimodal_clinical as clinical  # noqa: E402
from download_images import load_checksums  # noqa: E402
from download_mimic_iv_ed import login  # noqa: E402

BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
local = threading.local()


def session_for(cookies):
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.cookies.update(cookies)
    return local.session


class FetchError(Exception):
    """Failure category only (no URL, path or identifier), safe to print in aggregate."""


def fetch(name, target, expected, cookies, retries=4):
    partial = target.with_name(f"{target.name}.{threading.get_ident()}.part")
    for attempt in range(1, retries + 1):
        try:
            with session_for(cookies).get(BASE_URL + name, stream=True, timeout=(20, 120)) as response:
                if response.status_code in {401, 403}:
                    raise PermissionError(f"HTTP {response.status_code}; check MIMIC-CXR-JPG access")
                if response.status_code != 200:
                    raise FetchError(f"http_{response.status_code}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1 << 20):
                        stream.write(chunk)
            if clinical.sha256(partial) != expected:
                raise FetchError("checksum_mismatch")
            try:
                with Image.open(partial) as image:
                    image.load()
                    if image.format != "JPEG":
                        raise FetchError("not_jpeg")
            except OSError:
                raise FetchError("jpeg_decode_error") from None
            size = partial.stat().st_size
            partial.replace(target)
            return size
        except PermissionError:
            partial.unlink(missing_ok=True)
            raise
        except (requests.RequestException, FetchError, OSError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise error if isinstance(error, FetchError) else FetchError(type(error).__name__) from None
            time.sleep(2 * attempt)


def main(args):
    cohort = args.cohort.resolve()
    names = [line for line in (cohort / "download_list.txt").read_text().splitlines() if line]
    sums = load_checksums()
    for pattern in ("*.part", "*.jpg.partial"):  # leftovers of an interrupted parallel or sequential run
        for stale in cohort.rglob(pattern):
            stale.unlink()
    pending = []
    for name in names:
        clinical.require(name.startswith("files/") and ".." not in Path(name).parts, "Invalid selected image path")
        clinical.require(name in sums, "Selected image missing published checksum")
        target = cohort / name
        if not target.is_file() or clinical.sha256(target) != sums[name]:
            pending.append(name)
    print(json.dumps({"listed": len(names), "already_verified": len(names) - len(pending), "to_download": len(pending),
                      "workers": args.workers}), flush=True)
    if not pending or args.dry_run:
        return
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
    if not user or not password:
        user = user or input("PhysioNet username: ").strip()
        password = password or getpass.getpass("PhysioNet password (hidden, not saved): ")
    with requests.Session() as main_session:
        login(main_session, user, password)
        cookies = main_session.cookies
    started, done, failed, total_bytes, reasons = time.time(), 0, 0, 0, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, n, cohort / n, sums[n], cookies): n for n in pending}
        for future in as_completed(futures):
            try:
                total_bytes += future.result()
                done += 1
            except PermissionError as error:
                pool.shutdown(cancel_futures=True)
                raise SystemExit(str(error)) from None
            except Exception as error:
                failed += 1
                reason = str(error) if isinstance(error, FetchError) else type(error).__name__
                reasons[reason] = reasons.get(reason, 0) + 1
            if (done + failed) % 50 == 0 or done + failed == len(pending):
                elapsed = time.time() - started
                rate = total_bytes / max(elapsed, 1e-6) / 1e6
                eta = (len(pending) - done - failed) / max(done / max(elapsed, 1e-6), 1e-6) / 60
                print(json.dumps({"verified_downloads": done, "failed": failed, "required": len(pending),
                                  "MB_per_s": round(rate, 1), "eta_minutes": round(eta, 1)}), flush=True)
    print(json.dumps({"all_selected_images_available": failed == 0, "downloaded": done, "failed": failed,
                      "failure_reasons": reasons, "minutes": round((time.time() - started) / 60, 1),
                      "next": "re-run the same command to retry failures" if failed else "done"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Only count verified/pending files; no network")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        parser.error("Use 1-16 workers")
    main(arguments)
