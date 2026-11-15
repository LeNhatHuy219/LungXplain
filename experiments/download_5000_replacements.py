"""Download the twelve selected source images using the user's PhysioNet access.

Run locally in Terminal; password entry is hidden and never saved. Existing
PHYSIONET_USER/PHYSIONET_PASSWORD environment variables are also supported.
Only aggregate progress is printed, without patient IDs or source image paths.
"""
from __future__ import annotations

import argparse
import getpass
from html.parser import HTMLParser
import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_multimodal_clinical as clinical
from download_images import load_checksums

ROOT = clinical.ROOT
DEFAULT_COHORT = ROOT / "data/mimic_cxr/cohort_5000"
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"


class LoginForms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current = {"action": attrs.get("action", ""), "method": attrs.get("method", "get"), "inputs": []}
            self.forms.append(self.current)
        if tag == "input" and self.current is not None:
            self.current["inputs"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


def download(args):
    if getattr(args, "inspect_login", False):
        response = requests.get("https://physionet.org/login/", timeout=(20, 60))
        parser = LoginForms()
        parser.feed(response.text)
        print(json.dumps({"status": response.status_code,
                          "forms": [{"action": form["action"], "method": form["method"],
                                     "fields": [{"name": item.get("name", ""), "type": item.get("type", "text")} for item in form["inputs"]]}
                                    for form in parser.forms]}), flush=True)
        return
    names = [line for line in (args.cohort / "download_list.txt").read_text().splitlines() if line]
    sums = load_checksums()
    pending = []
    for name in names:
        target = args.cohort / name
        clinical.require(name.startswith("files/") and ".." not in Path(name).parts, "Invalid selected image path")
        clinical.require(name in sums, "Selected image missing published checksum")
        if not target.is_file() or clinical.sha256(target) != sums[name]:
            pending.append(name)
    if not pending:
        print(json.dumps({"all_selected_images_available": True, "downloaded": 0, "already_verified": len(names)}))
        return
    user = os.environ.get("PHYSIONET_USER")
    password = os.environ.get("PHYSIONET_PASSWORD")
    if not user or not password:
        if args.non_interactive:
            raise SystemExit("PhysioNet authentication required. Run this script in your Terminal; do not paste credentials into chat.")
        user = user or input("PhysioNet username: ").strip()
        password = password or getpass.getpass("PhysioNet password (hidden, not saved): ")
    clinical.require(bool(user and password), "Credentials must not be empty")
    completed = 0
    with requests.Session() as session:
        if getattr(args, "session_login", False):
            login = session.get("https://physionet.org/login/", timeout=(20, 60))
            parser = LoginForms()
            parser.feed(login.text)
            form = next((form for form in parser.forms if {"username", "password"}
                         <= {item.get("name") for item in form["inputs"]}), None)
            clinical.require(form is not None, "PhysioNet login form was not found")
            action = urljoin(login.url, form["action"])
            clinical.require(urlparse(action).hostname == "physionet.org" and action.startswith("https://"),
                             "Unexpected authentication destination")
            payload = {item["name"]: item.get("value", "") for item in form["inputs"]
                       if item.get("name") and item.get("type") == "hidden"}
            payload.update(username=user, password=password)
            response = session.post(action, data=payload, headers={"Referer": login.url},
                                    timeout=(20, 60), allow_redirects=False)
            body = response.text.lower()
            rejected = "enter a correct" in body or "incorrect password" in body or "invalid username" in body
            print(json.dumps({"web_login_status": response.status_code, "authentication_rejected": rejected,
                              "redirect_path": urlparse(response.headers.get("Location", "")).path}), flush=True)
            clinical.require(not rejected, "PhysioNet rejected the supplied username/password")
            clinical.require(response.status_code in {302, 303}, "PhysioNet web login requires an additional user step")
        else:
            session.auth = (user, password)
        if getattr(args, "diagnose_auth", False):
            for label, url in (("project_readme", BASE_URL + "README"),
                               ("project_page", "https://physionet.org/content/mimic-cxr-jpg/2.1.0/")):
                response = session.get(url, timeout=(20, 60))
                body = response.text.lower() if "text" in response.headers.get("Content-Type", "") else ""
                print(json.dumps({"resource": label, "status": response.status_code,
                                  "content_type": response.headers.get("Content-Type", ""),
                                  "proxy_block_reason": response.headers.get("X-Mitmproxy-Blocked-Reason", ""),
                                  "notice_terms": {term: term in body for term in
                                                   ("credentialed", "permission", "forbidden", "access denied", "incorrect", "log in", "robots_denied")}},
                                 ensure_ascii=False), flush=True)
            return
        for name in pending:
            target = args.cohort / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".jpg.partial")
            try:
                with session.get(BASE_URL + name, stream=True, timeout=(20, 60)) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"PhysioNet returned HTTP {response.status_code}; check account access to MIMIC-CXR-JPG 2.1.0")
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                clinical.require(clinical.sha256(partial) == sums[name], "Downloaded image checksum mismatch")
                with Image.open(partial) as image:
                    image.load()
                    clinical.require(image.format == "JPEG", "Downloaded file is not a source JPEG")
                partial.replace(target)
            except requests.RequestException as error:
                partial.unlink(missing_ok=True)
                raise SystemExit("PhysioNet connection failed (" + type(error).__name__ + "); no source path or credentials are logged.") from None
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            completed += 1
            print(json.dumps({"verified_downloads": completed, "required_downloads": len(pending)}), flush=True)
    print(json.dumps({"all_selected_images_available": True, "downloaded": completed, "already_verified": len(names) - len(pending)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--diagnose-auth", action="store_true")
    parser.add_argument("--inspect-login", action="store_true")
    parser.add_argument("--session-login", action="store_true", default=True)
    parser.add_argument("--basic-auth", dest="session_login", action="store_false")
    download(parser.parse_args())
