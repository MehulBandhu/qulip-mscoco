"""Figure revision: stop fitting one law through two behaviours.

The single power law in parameter count was averaging over an interaction. With
n=2 L=5 included, the residuals stop being scattered and start sorting by shape:
every three-qubit configuration above two layers sits below the line and every
two-qubit configuration above three layers sits above it. Per axis, depth at two
qubits is 0.58 per parameter, depth at three qubits is 0.37, and width at two
layers is 0.52. Those are different behaviours and one exponent hides them.

So F1 fits only the width series, where the exponent means something, and shows
the depth series as measured. The leave-one-out spread on the old nine-point fit
was 0.459 to 0.538, and L=6 is still running, so a single headline exponent
would move again anyway.

Also here: confidence intervals on the ARO panel, which at 3,650 rows has a
standard error of 0.0083 and was reading as "roughly chance" when the grammar
model is 2.9 SE below on attributes and 4.9 SE above on relations; the
concentration annotation moved out of the data; and the data-scaling figure
restored as supplementary with its epoch counts shown, since cutting it silently
loses the crossover.

    python figures_v4.py
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
from matplotlib import cm

REPORT = Path("report")
FIGS = REPORT / "figures"
SINGLE, DOUBLE = 3.5, 7.2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "grid.alpha": 0.3, "grid.linestyle": "--", "grid.linewidth": 0.5,
})
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
CONFIG = re.compile(r"(\d+) qubits?/type, (\d+) layers$")

ARO_ROWS = 3650                     # attribute and relation splits alike
ARO = {"grammar": (0.476, 0.540), "bag of words": (0.311, 0.292)}
AT_20K = {"grammar": dict(r1=0.92, r5=4.22, r10=7.60, params=479_353),
          "bag of words": dict(r1=6.44, r5=20.58, r10=31.00, params=1_329_967)}
# Training-set sweep. Epochs differ, which is why nothing is fitted through it.
DATA_SWEEP = {
    "quantum, 1.2M parameters": {5000: (0.12, 25), 10000: (0.70, 29),
                                 20000: (0.92, 36), 118287: (3.08, 99)},
    "classical, up to 2.3B": {5000: (0.52, 30), 10000: (0.78, 30),
                              20000: (1.00, 30), 118287: (1.72, 30)},
}
CONCENTRATION = {1: (2.93, 1.038e-4), 2: (5.02, 9.978e-5), 3: (7.09, 9.810e-5),
                 4: (9.13, 9.983e-5), 5: (11.18, 9.933e-5)}


def load(name):
    rows = []
    for r in csv.DictReader(open(REPORT / name)):
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
        rows.append(row)
    return rows


def save(fig, stem):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  figures/{stem}.pdf")


def panel(ax, letter, dx=-0.22):
    ax.text(dx, 1.04, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")


def power_fit(x, y, n_boot=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    rng = np.random.default_rng(0)
    reps = np.array([np.polyfit(np.log(x[i]), np.log(y[i]), 1)
                     for i in (rng.integers(0, len(x), len(x))
                               for _ in range(n_boot)) if len(set(i)) > 1])
    pred = np.exp(loga) * x ** b
    ss = np.sum((np.log(y) - np.log(pred)) ** 2)
    st = np.sum((np.log(y) - np.log(y).mean()) ** 2)
    return dict(b=b, a=np.exp(loga), reps=reps, n=len(x),
                lo=np.percentile(reps[:, 0], 2.5),
                hi=np.percentile(reps[:, 0], 97.5), r2=1 - ss / st)


def text_on(value, vmin, vmax):
    """Pick black or white by the colormap's luminance, not by a fraction of
    the maximum, which flips at the wrong place whenever the range moves."""
    r, g, b, _ = cm.viridis((value - vmin) / (vmax - vmin))
    return "k" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "w"


def main():
    test, train = load("test_results.csv"), load("training_curves.csv")

    finals = {}
    for r in test:
        if not r.get("i2tR1") or (r.get("epoch") or 0) < 80:
            continue
        lab = r.get("label") or ""
        if lab not in finals or r["epoch"] > finals[lab]["epoch"]:
            finals[lab] = r

    params = {}
    for r in train:
        m = CONFIG.match(r.get("label") or "")
        if m and r.get("text_params"):
            params[(int(m.group(1)), int(m.group(2)))] = int(r["text_params"])
    for k, p in (((3, 2), 2_681_521), ((4, 2), 3_444_019), ((1, 2), 1_156_525)):
        params.setdefault(k, p)

    grid = {}
    for lab, r in finals.items():
        m = CONFIG.match(lab)
        if m:
            grid[(int(m.group(1)), int(m.group(2)))] = r["i2tR1"]
    classical = next((r["i2tR1"] for lab, r in finals.items()
                      if re.match(r"classical, full", lab, re.I)), 1.72)

    # ---- F1: fit the width series only, show depth as measured ----------
    width = sorted((params[(n, 2)], grid[(n, 2)], n) for n in range(1, 6)
                   if (n, 2) in grid and (n, 2) in params)
    fig, ax = plt.subplots(figsize=(SINGLE, 3.2))
    if len(width) >= 3:
        f = power_fit([w[0] for w in width], [w[1] for w in width])
        g = np.logspace(np.log10(width[0][0]), np.log10(width[-1][0]), 80)
        curves = np.array([np.exp(c[1]) * g ** c[0] for c in f["reps"]])
        ax.fill_between(g / 1e6, np.percentile(curves, 2.5, axis=0),
                        np.percentile(curves, 97.5, axis=0),
                        color=OKABE[0], alpha=0.15, lw=0, zorder=1)
        ax.plot(g / 1e6, f["a"] * g ** f["b"], color=OKABE[0], lw=1.2,
                ls="--", zorder=2,
                label=f"width, 2 layers: $N^{{{f['b']:.2f}}}$")
        print(f"    width at 2 layers: {f['b']:.3f} "
              f"[{f['lo']:.3f}, {f['hi']:.3f}], R2 {f['r2']:.3f}")
    ax.plot([w[0] / 1e6 for w in width], [w[1] for w in width], "o",
            color=OKABE[0], ms=6, zorder=4)

    for i, n in enumerate((2, 3), start=1):
        sel = sorted((params[(n, l)], grid[(n, l)], l) for l in range(2, 7)
                     if (n, l) in grid and (n, l) in params)
        if len(sel) >= 2:
            ax.plot([s[0] / 1e6 for s in sel], [s[1] for s in sel],
                    marker=MARKERS[i], ls=STYLES[i], color=OKABE[i], ms=6,
                    zorder=3, label=f"depth, {n} qubits per type")
            if len(sel) >= 3:
                fd = power_fit([s[0] for s in sel], [s[1] for s in sel], 800)
                print(f"    depth at {n} qubits/type: {fd['b']:.3f} "
                      f"[{fd['lo']:.3f}, {fd['hi']:.3f}]")
    ax.axhline(classical, color="0.45", lw=1.0, ls=":")
    ax.text(0.5, classical * 1.10, "classical tensor network, 2.3B parameters",
            transform=ax.get_yaxis_transform(), ha="center",
            fontsize=7, color="0.45")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(classical * 0.72, max(grid.values()) * 1.6)
    ax.set_xlabel("Text encoder parameters (millions)")
    ax.set_ylabel("Correct caption ranked first (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.grid(which="both")
    save(fig, "F1_model_size")

    # ---- F2: grid and the same points against parameters ----------------
    ns, ls_ = sorted({k[0] for k in grid}), sorted({k[1] for k in grid})
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))
    m = np.full((len(ns), len(ls_)), np.nan)
    for (n, l), v in grid.items():
        m[ns.index(n)][ls_.index(l)] = v
    lo, hi = np.nanmin(m), np.nanmax(m)
    im = ax[0].imshow(m, cmap="viridis", aspect="auto", origin="lower")
    ax[0].set_xticks(range(len(ls_)), [str(l) for l in ls_])
    ax[0].set_yticks(range(len(ns)), [str(n) for n in ns])
    for i in range(len(ns)):
        for j in range(len(ls_)):
            if not np.isnan(m[i][j]):
                ax[0].text(j, i, f"{m[i][j]:.2f}", ha="center", va="center",
                           fontsize=9, color=text_on(m[i][j], lo, hi))
    ax[0].set_xlabel("Ansatz layers")
    ax[0].set_ylabel("Qubits per grammatical type")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04
                 ).set_label("Correct caption ranked first (%)", fontsize=9)
    panel(ax[0], "a", dx=-0.24)

    for i, n in enumerate(ns):
        sel = sorted((params[(n, l)], grid[(n, l)]) for l in ls_
                     if (n, l) in grid and (n, l) in params)
        if len(sel) >= 2:
            ax[1].plot([p / 1e6 for p, _ in sel], [v for _, v in sel],
                       marker=MARKERS[i % len(MARKERS)],
                       ls=STYLES[i % len(STYLES)], color=OKABE[i % len(OKABE)],
                       label=f"{n} qubits per type, varying depth")
    sel = sorted((params[(n, 2)], grid[(n, 2)]) for n in ns
                 if (n, 2) in grid and (n, 2) in params)
    if len(sel) >= 2:
        ax[1].plot([p / 1e6 for p, _ in sel], [v for _, v in sel], marker="X",
                   ls=(0, (1, 1)), color=OKABE[5], lw=2,
                   label="2 layers, varying width")
    ax[1].set_xlabel("Text encoder parameters (millions)")
    ax[1].set_ylabel("Correct caption ranked first (%)")
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    ax[1].grid()
    ax[1].text(0.98, 0.02, "two-layer points appear in both sweeps",
               transform=ax[1].transAxes, fontsize=7, color="0.4",
               ha="right", va="bottom")
    panel(ax[1], "b", dx=-0.20)
    fig.tight_layout()
    save(fig, "F2_depth_and_width")

    # ---- F3: dynamics ----------------------------------------------------
    runs = defaultdict(list)
    for r in train:
        if CONFIG.match(r.get("label") or "") and r.get("loss"):
            runs[r["label"]].append(r)
    wanted = ["1 qubit/type, 2 layers", "2 qubits/type, 2 layers",
              "2 qubits/type, 4 layers", "2 qubits/type, 5 layers"]
    keep = {k: sorted(runs[k], key=lambda x: x["epoch"])
            for k in wanted if len(runs.get(k, [])) > 80}
    if keep:
        fig, ax = plt.subplots(2, 2, figsize=(DOUBLE, 5.4))
        spec = [("loss", "Training loss", ax[0][0], "a"),
                ("val_i2t_r10", "Correct caption in top 10 (%)", ax[0][1], "b"),
                ("grad_norm", "Gradient norm", ax[1][0], "c"),
                ("gamma", "Separation of true pairs", ax[1][1], "d")]
        for i, (label, rows) in enumerate(keep.items()):
            for key, _, a, _ in spec:
                xy = [(r["epoch"], r[key]) for r in rows
                      if r.get(key) is not None]
                if xy:
                    s = 100 if key.startswith("val") else 1
                    a.plot([p[0] for p in xy], [p[1] * s for p in xy],
                           color=OKABE[i], ls=STYLES[i], lw=1.3, label=label)
        for _, ylab, a, letter in spec:
            a.set_xlabel("Epoch"); a.set_ylabel(ylab); a.grid()
            panel(a, letter, dx=-0.19)
        ax[0][0].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        save(fig, "F3_training_dynamics")

    # ---- F4: the ablation, with intervals on the ARO panel ---------------
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))
    names = list(AT_20K)
    x = np.arange(len(names))
    for i, (key, lab) in enumerate((("r1", "top 1"), ("r5", "top 5"),
                                    ("r10", "top 10"))):
        ax[0].bar(x + (i - 1) * 0.26, [AT_20K[n][key] for n in names], 0.26,
                  color=OKABE[i], label=lab)
    ax[0].set_xticks(x, [f"{n}\n({AT_20K[n]['params'] / 1e6:.2f}M parameters)"
                         for n in names], fontsize=8.5)
    ax[0].set_ylabel("Correct caption found (%)")
    ax[0].set_title("Retrieval, 20,000 training images", fontsize=9.5)
    ax[0].legend(frameon=False); ax[0].grid(axis="y")
    panel(ax[0], "a", dx=-0.20)

    # A dot plot with 95% intervals, so the truncated axis cannot flatter the
    # grammar model into looking like it sits at chance.
    se = np.sqrt(0.25 / ARO_ROWS)
    for i, (name, (att, rel)) in enumerate(ARO.items()):
        for j, (v, lab) in enumerate(((att, "attributes"), (rel, "relations"))):
            y = i * 2 + j * 0.6
            ax[1].errorbar(v, y, xerr=1.96 * se, fmt=MARKERS[j],
                           color=OKABE[j + 3], ms=6, capsize=3, lw=1.2)
            ax[1].text(v, y + 0.22, f"{(v - 0.5) / se:+.1f} SE",
                       ha="center", fontsize=7, color="0.35")
    ax[1].axvline(0.5, color="0.35", lw=1.0, ls="--")
    ax[1].text(0.5, -0.85, "chance", ha="center", fontsize=8, color="0.35")
    ax[1].set_yticks([0.3, 2.3], names, fontsize=8.5)
    ax[1].set_ylim(-1.1, 3.2)
    ax[1].set_xlim(0.25, 0.60)
    ax[1].set_xlabel("Correct caption chosen (fraction)")
    ax[1].set_title("Word order, zero-shot on ARO", fontsize=9.5)
    ax[1].grid(axis="x")
    for j, lab in enumerate(("attributes", "relations")):
        ax[1].plot([], [], MARKERS[j], color=OKABE[j + 3], label=lab)
    ax[1].legend(frameon=False, loc="lower right", fontsize=8)
    panel(ax[1], "b", dx=-0.20)
    fig.tight_layout()
    save(fig, "F4_grammar_ablation")

    # ---- F5: concentration, annotation clear of the data -----------------
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    w = [CONCENTRATION[n][0] for n in sorted(CONCENTRATION)]
    v = [CONCENTRATION[n][1] for n in sorted(CONCENTRATION)]
    ax.plot(w, v, marker="o", color=OKABE[0], ms=6, zorder=3, label="measured")
    for n in sorted(CONCENTRATION):
        ax.annotate(f"n={n}", CONCENTRATION[n], fontsize=8,
                    textcoords="offset points", xytext=(0, 9), ha="center")
    gw = np.linspace(min(w), max(w), 60)
    ax.plot(gw, v[0] * 2.0 ** (-(gw - w[0])), color="0.45", lw=1.1, ls="--",
            zorder=2, label=r"$2^{-w}$, barren plateau")
    ax.set_yscale("log")
    ax.set_ylim(min(v) * 3e-4, max(v) * 4)
    ax.set_xlabel("Mean qubits per word register")
    ax.set_ylabel("Variance of the loss at initialisation")
    ax.legend(frameon=False, loc="center left", fontsize=8)
    ax.grid(which="both")
    ax.text(0.97, 0.06,
            f"measured {v[0] / v[-1]:.2f}$\\times$\n"
            f"predicted {2.0 ** (w[-1] - w[0]):.0f}$\\times$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    save(fig, "F5_loss_spread")

    # ---- F6: training-set size, supplementary, nothing fitted ------------
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    for i, (lab, d) in enumerate(DATA_SWEEP.items()):
        xs = sorted(d)
        ax.plot(xs, [d[x][0] for x in xs], marker=MARKERS[i], ls=STYLES[i],
                color=OKABE[i], ms=6, label=lab)
        for x in xs:
            ax.annotate(f"{d[x][1]}ep", (x, d[x][0]), fontsize=6.5,
                        color="0.4", textcoords="offset points",
                        xytext=(0, -11), ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training images")
    ax.set_ylabel("Correct caption ranked first (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(which="both")
    ax.text(0.97, 0.05, "epochs differ between points;\nno slope is fitted",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="0.4")
    save(fig, "F6_training_set_size")


if __name__ == "__main__":
    main()
