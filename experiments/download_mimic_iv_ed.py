"""Download the MIMIC-IV-ED tables needed for dataset v5 with the user's PhysioNet access.

Run in your own Terminal; the password prompt is hidden and nothing is saved.
PHYSIONET_USER/PHYSIONET_PASSWORD environment variables are also accepted. Every
file is checked against the project's published SHA256SUMS.txt before it is kept.
Requires credentialed access and a signed data use agreement for MIMIC-IV-ED.
Only aggregate progress is printed.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_5000_replacements import LoginForms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ("ed/edstays.csv.gz", "ed/triage.csv.gz", "ed/vitalsign.csv.gz")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def login(session, user, password):
    page = session.get("https://physionet.org/login/", timeout=(20, 60))
    parser = LoginForms()
    parser.feed(page.text)
    form = next((f for f in parser.forms if {"username", "password"} <= {i.get("name") for i in f["inputs"]}), None)
    if form is None:
        raise SystemExit("PhysioNet login form was not found")
    action = urljoin(page.url, form["action"])
    if urlparse(action).hostname != "physionet.org" or not action.startswith("https://"):
        raise SystemExit("Unexpected authentication destination")
    payload = {i["name"]: i.get("value", "") for i in form["inputs"] if i.get("name") and i.get("type") == "hidden"}
    payload.update(username=user, password=password)
    response = session.post(action, data=payload, headers={"Referer": page.url}, timeout=(20, 60), allow_redirects=False)
    body = response.text.lower()
    if "enter a correct" in body or "incorrect password" in body or "invalid username" in body:
        raise SystemExit("PhysioNet rejected the supplied username/password")
    if response.status_code not in {302, 303}:
        raise SystemExit("PhysioNet web login requires an additional user step")


def main(args):
    base = f"https://physionet.org/files/mimic-iv-ed/{args.version}/"
    destination = (args.dest or ROOT / f"mimic-iv-ed-{args.version}").resolve()
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
    if not user or not password:
        user = user or input("PhysioNet username: ").strip()
        password = password or getpass.getpass("PhysioNet password (hidden, not saved): ")
    with requests.Session() as session:
        login(session, user, password)
        sums_response = session.get(base + "SHA256SUMS.txt", timeout=(20, 60))
        if sums_response.status_code != 200:
            raise SystemExit(f"HTTP {sums_response.status_code} for MIMIC-IV-ED {args.version}; check project access/DUA")
        sums = {}
        for line in sums_response.text.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                sums[parts[1].strip().lstrip("*")] = parts[0]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "SHA256SUMS.txt").write_text(sums_response.text)
        for name in FILES:
            if name not in sums:
                raise SystemExit(f"{name} is not listed in SHA256SUMS.txt for version {args.version}")
            target = destination / name
            if target.is_file() and sha256(target) == sums[name]:
                print(json.dumps({"file": name, "status": "already_verified"}), flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            try:
                with session.get(base + name, stream=True, timeout=(20, 120)) as response:
                    if response.status_code != 200:
                        raise SystemExit(f"HTTP {response.status_code} for {name}; check MIMIC-IV-ED access")
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1 << 20):
                            stream.write(chunk)
                if sha256(partial) != sums[name]:
                    raise SystemExit(f"Checksum mismatch for {name}; file discarded")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            print(json.dumps({"file": name, "status": "downloaded_and_verified", "bytes": target.stat().st_size}), flush=True)
    print(json.dumps({"mimic_iv_ed_ready": True, "version": args.version, "files": len(FILES)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.2")
    parser.add_argument("--dest", type=Path, default=None)
    main(parser.parse_args())
