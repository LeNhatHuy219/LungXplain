"""Paper figure: validation intervention curves of the main ACC-ECBM configuration.

Aggregates the saved per-seed ``interventions_validation.csv`` (mean ± SD over seeds) and
writes a vector PDF to paper/lncs/figures/. Only aggregate curves are plotted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "calibrated_intervention_aware"
# Validated categorical slots 1-3 (fixed order); markers give a secondary encoding.
STYLE = {"no_correction": ("No correction", "#2a78d6", "o"),
         "hard_correction": ("Hard correction", "#eb6834", "s"),
         "acc": ("ACC", "#1baf7a", "^")}
PANELS = [("remaining_concept_logloss", "Unconfirmed-concept log-loss"),
          ("auroc", "Mortality AUROC"),
          ("logloss", "Mortality log-loss")]


def main(args):
    frames = []
    for path in sorted(args.runs.rglob(f"{VARIANT}/interventions_validation.csv")):
        table = pd.read_csv(path)
        table["seed"] = path.parent.parent.name
        frames.append(table)
    if len(frames) < 2:
        raise RuntimeError("Need at least two completed seeds")
    data = pd.concat(frames)
    data = data[data.requested_concepts <= args.max_m]
    grouped = data.groupby(["strategy", "requested_concepts"])
    mean, sd = grouped.mean(numeric_only=True), grouped.std(numeric_only=True)
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(4.8, 1.85), constrained_layout=True)
    for axis, (metric, title) in zip(axes, PANELS):
        for strategy, (label, color, marker) in STYLE.items():
            if metric == "remaining_concept_logloss" and strategy == "no_correction":
                continue  # identical to hard correction on unconfirmed concepts
            m = mean.loc[strategy][metric].dropna()
            s = sd.loc[strategy][metric].reindex(m.index)
            axis.fill_between(m.index, m - s, m + s, color=color, alpha=0.12, linewidth=0)
            axis.plot(m.index, m.values, color=color, linewidth=1.2, marker=marker, markersize=2.8,
                      markeredgecolor="#fcfcfb", markeredgewidth=0.5, label=label)
        axis.set_title(title, color="#0b0b0b", loc="left")
        axis.set_xlabel("Requested $m$")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.5)
        axis.set_xticks(range(0, args.max_m + 1, 1))
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print({"figure": str(args.output.relative_to(ROOT)), "seeds": len(frames), "max_m": args.max_m})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "data/kaggle_v5_output/native_output")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/lncs/figures/interventions_validation.pdf")
    parser.add_argument("--max-m", type=int, default=6)
    main(parser.parse_args())
