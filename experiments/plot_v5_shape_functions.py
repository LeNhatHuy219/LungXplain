"""Paper figure: EBM shape functions of the main ACC-ECBM configuration (validation-frozen).

Plots the learned main-effect contribution (log-odds) of four terms for every seed. Only
model parameters are used (bin cut points and term scores). The outermost bin edges,
which equal the training minimum and maximum, are not drawn, so no individual value is
shown. Writes a vector PDF to paper/lncs/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "calibrated_intervention_aware"
TERMS = [("Support Devices", "Support Devices (prob.)"), ("Edema", "Edema (prob.)"),
         ("clinical_age_years_approx", "Age (years)"), ("clinical_wbc", "WBC (K/\u00b5L)")]
COLOR = "#2a78d6"  # single series (seeds share one hue); first validated categorical slot
SEED_STYLE = ["-", "--", ":"]


def main(args):
    models = sorted(args.runs.rglob(f"{VARIANT}/ebm.joblib"),
                    key=lambda p: int(p.parent.parent.name.removeprefix("seed")))
    if len(models) < 2:
        raise RuntimeError("Need at least two completed seeds")
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, len(TERMS), figsize=(4.8, 1.55), constrained_layout=True)
    for k, path in enumerate(models):
        ebm = joblib.load(path)
        explanation = ebm.explain_global()
        seed = path.parent.parent.name.removeprefix("seed")
        for axis, (term, _) in zip(axes, TERMS):
            data = explanation.data(list(ebm.term_names_).index(term))
            edges, scores = np.asarray(data["names"], float), np.asarray(data["scores"], float)
            inner = edges[1:-1]  # drop training min/max
            x = np.r_[inner[0], np.repeat(inner[1:-1], 2), inner[-1]]
            y = np.repeat(scores[1:-1], 2)
            axis.plot(x, y, color=COLOR, linewidth=1.0, linestyle=SEED_STYLE[k % 3], label=f"Seed {seed}")
    for axis, (_, title) in zip(axes, TERMS):
        axis.axhline(0, color="#a8a7a2", linewidth=0.5)
        axis.set_title(title, color="#0b0b0b", loc="left")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.5)
    axes[0].set_ylabel("Contribution (log-odds)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=len(models), frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print({"figure": str(args.output.relative_to(ROOT)), "seeds": len(models), "terms": len(TERMS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "data/kaggle_v5_output/native_output")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/lncs/figures/shape_functions.pdf")
    main(parser.parse_args())
