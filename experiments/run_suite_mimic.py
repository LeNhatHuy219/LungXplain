#!/usr/bin/env python3
"""
Automated Multi-Seed Suite Runner for MIMIC-CXR mortality prediction.
Sequentially trains across 3 seeds (42, 123, 2026), evaluates, and aggregates results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 123, 2026]


def run_cmd(cmd: list[str]):
    print(f"\n[Suite Runner] Launching: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[Suite Runner] WARNING: Command exited with code {res.returncode}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["blackbox", "soft", "hard", "ecbm", "all"], default="soft")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--manifest-path", type=Path, default=ROOT / "data/mimic_cxr/manifest_mortality.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mimic_cxr")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    models_to_run = ["blackbox", "soft", "hard", "ecbm"] if args.model == "all" else [args.model]

    for m in models_to_run:
        for seed in args.seeds:
            out_dir = ROOT / f"results/mimic_cxr/{m}/seed{seed}_e{args.epochs}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if m == "blackbox":
                cmd = [
                    sys.executable, "-u", str(ROOT / "experiments/run_blackbox_mimic.py"),
                    "--manifest-path", str(args.manifest_path),
                    "--data-root", str(args.data_root),
                    "--epochs", str(args.epochs),
                    "--seed", str(seed),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                    "--output-dir", str(out_dir),
                ]
            elif m in ("soft", "hard"):
                cmd = [
                    sys.executable, "-u", str(ROOT / "experiments/run_cbm_mimic.py"),
                    "--model-type", m,
                    "--manifest-path", str(args.manifest_path),
                    "--data-root", str(args.data_root),
                    "--epochs", str(args.epochs),
                    "--seed", str(seed),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                    "--output-dir", str(out_dir),
                ]
            elif m == "ecbm":
                cmd = [
                    sys.executable, "-u", str(ROOT / "experiments/run_ecbm_mimic.py"),
                    "--manifest-path", str(args.manifest_path),
                    "--data-root", str(args.data_root),
                    "--epochs", str(args.epochs),
                    "--seed", str(seed),
                    "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--device", args.device,
                    "--output-dir", str(out_dir),
                ]
                    "--device", args.device,
                    "--output-dir", str(out_dir),
                ]

            print(f"\n{'='*70}")
            print(f"RUNNING MODEL: {m.upper()} | SEED: {seed} | EPOCHS: {args.epochs}")
            print(f"{'='*70}")
            run_cmd(cmd)

    # Automatically aggregate seeds at the end
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS ACROSS SEEDS...")
    print(f"{'='*70}")
    agg_cmd = [
        sys.executable, str(ROOT / "experiments/aggregate_seeds.py"),
        "--seeds", *[str(s) for s in args.seeds]
    ]
    run_cmd(agg_cmd)


if __name__ == "__main__":
    main()
