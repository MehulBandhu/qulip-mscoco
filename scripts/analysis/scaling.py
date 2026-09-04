"""Scaling analysis: fit power laws in parameters, depth, width and data.

Reads report/test_results.csv and report/training_curves.csv from
compile_results.py. Writes report/scaling.md and report/fig_scaling_*.pdf.

Every fit is bootstrapped over the points, so the confidence intervals reflect
how few configurations there are rather than pretending otherwise. With one run
per configuration these are descriptive, not inferential - seeds are what would
make them inferential, and the report says so.

    python scaling_analysis.py
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path("report")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
CB = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494"]

# Data-scaling points, measured at one qubit per type and two layers.
DATA_SCALING = {5000: 0.12, 10000: 0.70, 20000: 0.92, 118287: 3.08}

# Reference points from the literature and the classical baseline.
BARS = {"best tensor network": 1.26, "classical grammar": 1.72,
        "zero-shot CLIP": 48.4}


def load(name):
    p = REPORT / name
    if not p.exists():
        raise SystemExit(f"{p} missing - run compile_results.py first")
    out = []
    for r in csv.DictReader(open(p)):
        row = {}
        for k, v in r.items():
            if v in ("", None):
                row[k] = None
            elif k in ("run", "label", "source"):
                row[k] = v
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
        out.append(row)
    return out


def power_fit(x, y, n_boot=2000, seed=0):
    """Fit y = a * x^b in log space, with a bootstrap interval on b."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return None
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(set(idx)) < 2:
            continue
        bb, _ = np.polyfit(np.log(x[idx]), np.log(y[idx]), 1)
        bs.append(bb)
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    pred = np.exp(loga) * x ** b
    ss_res = np.sum((np.log(y) - np.log(pred)) ** 2)
    ss_tot = np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return dict(exponent=b, amplitude=np.exp(loga), lo=lo, hi=hi,
                r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan, n=len(x))


def converged(test, min_epoch=80):
    """Best benchmark per configuration, restricted to runs near their cap."""
    best = {}
    for r in test:
        if not r.get("i2tR1") or not r.get("epoch") or r["epoch"] < min_epoch:
            continue
        total = r["i2tR1"] + (r.get("i2tR5") or 0) + (r.get("i2tR10") or 0)
        if r["label"] not in best or total > best[r["label"]][0]:
            best[r["label"]] = (total, r)
    return {k: v[1] for k, v in best.items()}


def params_of(train):
    out = {}
    for r in train:
        if r.get("text_params"):
            out[r["label"]] = int(r["text_params"])
    return out


def main():
    test, train = load("test_results.csv"), load("training_curves.csv")
    pts, params = converged(test), params_of(train)
    lines = ["# Scaling", ""]

    # --- parameters ------------------------------------------------------
    xs, ys, labels = [], [], []
    for label, r in pts.items():
        if label in params:
            xs.append(params[label]); ys.append(r["i2tR1"]); labels.append(label)
    fit = power_fit(xs, ys)
    if fit:
        lines += ["## Recall against parameter count", "",
                  f"Fitting R@1 = a N^b over {fit['n']} converged configurations:",
                  "",
                  f"- exponent b = {fit['exponent']:.3f} "
                  f"(95% bootstrap {fit['lo']:.3f} to {fit['hi']:.3f})",
                  f"- R^2 in log space = {fit['r2']:.3f}", "",
                  "Every point is a single run, so this describes the "
                  "configurations measured rather than estimating a population "
                  "exponent. Repeat seeds would be needed for the latter.", ""]

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.scatter(np.array(xs) / 1e6, ys, s=24, color=CB[0], zorder=3)
        grid = np.linspace(min(xs), max(xs), 100)
        ax.plot(grid / 1e6, fit["amplitude"] * grid ** fit["exponent"],
                color=CB[0], lw=1, ls="--", alpha=0.6,
                label=f"$N^{{{fit['exponent']:.2f}}}$")
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x / 1e6, y), fontsize=6,
                        textcoords="offset points", xytext=(3, 3))
        for name, v in BARS.items():
            if v < max(ys) * 3:
                ax.axhline(v, color=CB[5], lw=0.7, ls=":")
                ax.text(0.02, v * 1.05, name, fontsize=6, color=CB[5],
                        transform=ax.get_yaxis_transform())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("text-encoder parameters (millions)")
        ax.set_ylabel("R@1 (%)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2, lw=0.5, which="both")
        fig.savefig(REPORT / "fig_scaling_params.pdf")
        fig.savefig(REPORT / "fig_scaling_params.png")
        plt.close(fig)

    # --- depth and width separately --------------------------------------
    grid = {}
    for label, r in pts.items():
        m = re.match(r"n=(\d+), L=(\d+)$", label)
        if m:
            grid[(int(m.group(1)), int(m.group(2)))] = r["i2tR1"]

    if len(grid) >= 4:
        lines += ["## Depth against width", "",
                  "| | " + " | ".join(f"L={l}" for l in
                                      sorted({k[1] for k in grid})) + " |",
                  "|---" * (len({k[1] for k in grid}) + 1) + "|"]
        for n in sorted({k[0] for k in grid}):
            row = [f"n={n}"] + [f"{grid.get((n, l), float('nan')):.2f}"
                                if (n, l) in grid else "-"
                                for l in sorted({k[1] for k in grid})]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        for axis, idx, other in (("layers", 1, 0), ("qubits per type", 0, 1)):
            slopes = []
            for fixed in sorted({k[other] for k in grid}):
                pairs = sorted((k[idx], v) for k, v in grid.items()
                               if k[other] == fixed)
                if len(pairs) >= 2:
                    f = power_fit([p[0] for p in pairs], [p[1] for p in pairs])
                    if f:
                        slopes.append(f["exponent"])
            if slopes:
                lines.append(f"- exponent in {axis}: "
                             + ", ".join(f"{s:.2f}" for s in slopes)
                             + f" (mean {np.mean(slopes):.2f})")
        lines.append("")

        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for i, n in enumerate(sorted({k[0] for k in grid})):
            pairs = sorted((k[1], v) for k, v in grid.items() if k[0] == n)
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-",
                        color=CB[i % len(CB)], lw=1.2, ms=4, label=f"n={n}")
        ax.set_xlabel("layers"); ax.set_ylabel("R@1 (%)")
        ax.legend(frameon=False); ax.grid(alpha=0.2, lw=0.5)
        fig.savefig(REPORT / "fig_scaling_depth.pdf")
        fig.savefig(REPORT / "fig_scaling_depth.png")
        plt.close(fig)

    # --- data ------------------------------------------------------------
    f = power_fit(list(DATA_SCALING), list(DATA_SCALING.values()))
    if f:
        lines += ["## Recall against training-set size", "",
                  "At one qubit per type and two layers:", "",
                  f"- exponent = {f['exponent']:.3f} "
                  f"(95% bootstrap {f['lo']:.3f} to {f['hi']:.3f})",
                  f"- R^2 in log space = {f['r2']:.3f}", ""]

        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        x = np.array(list(DATA_SCALING), float)
        ax.scatter(x, list(DATA_SCALING.values()), s=24, color=CB[2], zorder=3)
        gx = np.linspace(x.min(), x.max(), 100)
        ax.plot(gx, f["amplitude"] * gx ** f["exponent"], color=CB[2],
                lw=1, ls="--", alpha=0.6, label=f"$D^{{{f['exponent']:.2f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training images"); ax.set_ylabel("R@1 (%)")
        ax.legend(frameon=False); ax.grid(alpha=0.2, lw=0.5, which="both")
        fig.savefig(REPORT / "fig_scaling_data.pdf")
        fig.savefig(REPORT / "fig_scaling_data.png")
        plt.close(fig)

    # --- how many epochs to reach a given loss ---------------------------
    runs = defaultdict(list)
    for r in train:
        if r.get("loss"):
            runs[r["label"]].append((int(r["epoch"]), r["loss"]))
    target = 3.5
    reach = {}
    for label, rows in runs.items():
        hit = [e for e, l in sorted(rows) if l <= target]
        if hit and label in params:
            reach[label] = (params[label], min(hit))
    if len(reach) >= 3:
        f = power_fit([v[0] for v in reach.values()],
                      [v[1] for v in reach.values()])
        lines += [f"## Epochs to reach a training loss of {target}", "",
                  "| configuration | parameters | epochs |",
                  "|---|---|---|"]
        for label, (p, e) in sorted(reach.items(), key=lambda x: x[1][1]):
            lines.append(f"| {label} | {p:,} | {e} |")
        if f:
            lines += ["", f"- exponent in parameters: {f['exponent']:.3f} "
                          f"({f['lo']:.3f} to {f['hi']:.3f})"]
        lines.append("")

    (REPORT / "scaling.md").write_text("\n".join(lines) + "\n")
    print("  report/scaling.md")
    for p in sorted(REPORT.glob("fig_scaling_*.pdf")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
