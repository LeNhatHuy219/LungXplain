"""Build the resized pixel cache for a validated exact-5000 dataset (no learned features).

Uses the existing, hash-bound ``lungxplain.finetune.prepare_pixel_cache`` unchanged:
the cache index records the manifest hash, resize size, interpolation and the
dataset's image-audit hash, and every source JPEG is re-checked against the audit
before resizing. Refuses to run on a dataset without a passing validation report
(``load_cohort``). Output stays under the gitignored data/ directory; stdout is
aggregate only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungxplain.data import ROOT, load_cohort  # noqa: E402
from lungxplain.finetune import FineTuneConfig, prepare_pixel_cache  # noqa: E402


def main(args):
    output = args.output.resolve()
    output.relative_to(ROOT / "data")
    cohort = load_cohort(args.dataset)
    cache = prepare_pixel_cache(cohort, FineTuneConfig(resize_size=args.resize_size), output)
    index = json.loads((cache / "index.json").read_text())
    print(json.dumps({"pixel_cache_ready": True, "rows": index["rows"], "resize_size": args.resize_size,
                      "manifest_sha256": index["configuration"]["manifest_sha256"]}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize-size", type=int, default=256)
    main(parser.parse_args())
