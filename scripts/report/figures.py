"""Figures for the full-COCO run.

Reads the training logs and the saved benchmark files, so it can run any time
and picks up whatever has been measured so far. Nothing is recomputed.

    python figures.py

Writes PNGs into graphs/.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS, RESULTS, OUT = Path("logs"), Path("results"), Path("graphs")

BLUE, RED, GREEN, GREY = "#2a78d6", "#c0392b", "#1baf7a", "#8a8a8a"

EPOCH = re.compile(
    r"Epoch (\d+) \| Loss: ([\d.]+).*?i2tR1: ([\d.]+).*?i2tR5: ([\d.]+)"
    r".*?i2tR10: ([\d.]+).*?gamma: ([-\d.]+) \| i2t_hnm: ([-\d.]+)"
    r".*?(?:grad_norm: ([\d.e+-]+) \| )?Time: ([\d.]+)s")


def read_log(pattern):
    paths = sorted(LOGS.glob(pattern))
    if not paths:
        return []
    text = max(paths, key=lambda p: p.stat().st_size).read_text(
        errors="ignore").replace("\r", "\n")
    rows = []
    for line in text.splitlines():
        m = EPOCH.search(line)
        if m:
            e, loss, r1, r5, r10, gamma, hnm, grad, t = m.groups()
            rows.append(dict(epoch=int(e), loss=float(loss),
                             r1=float(r1) * 100, r5=float(r5) * 100,
                             r10=float(r10) * 100, gamma=float(gamma),
                             hnm=float(hnm),
                             grad=float(grad) if grad else None,
                             mins=float(t) / 60))
    return rows


def read_benchmarks():
    """Every (epoch, R@1, R@5, R@10) we have measured on the 5,000-image set."""
    pts = {}
    for f in list(RESULTS.glob("*.txt")) + list(LOGS.glob("q-bench*.out")):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        if "vqcfull" not in text and "vqcfull" not in f.name:
            continue
        ep = re.search(r"Recovered from Epoch: (\d+)", text)
        vals = re.findall(r"coco5k_i2tR(\d+)\s*\|\s*([\d.]+)", text)
        if ep and len(vals) >= 3:
            got = {int(k): float(v) * 100 for k, v in vals[:3]}
            pts[int(ep.group(1))] = (got.get(1), got.get(5), got.get(10))
    return dict(sorted(pts.items()))


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)


def main():
    OUT.mkdir(exist_ok=True)
    rows = read_log("q-full-fast-*.out")
    if not rows:
        raise SystemExit("no epoch lines found for the full-COCO run")
    print(f"  {len(rows)} epochs")
    ep = [r["epoch"] for r in rows]

    # 1. loss --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, [r["loss"] for r in rows], color=BLUE, lw=1.8)
    # Random guessing within a batch of 256 gives ln(256).
    ax.axhline(5.545, color=GREY, lw=0.9, ls=":")
    ax.text(len(ep) * 0.6, 5.60, "random guessing", fontsize=8, color=GREY)
    style(ax, "epoch", "training loss", "Training loss")
    fig.tight_layout(); fig.savefig(OUT / "01_loss.png", dpi=150); plt.close(fig)

    # 2. validation recall -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, c, lab in (("r10", BLUE, "top 10"), ("r5", RED, "top 5"),
                        ("r1", GREEN, "top 1")):
        ax.plot(ep, [r[key] for r in rows], color=c, lw=1.5, label=lab)
    style(ax, "epoch", "correct caption found (%)",
          "Validation recall, 1,000 images")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "02_validation.png", dpi=150); plt.close(fig)

    # 3. test set ----------------------------------------------------------
    bm = read_benchmarks()
    if bm:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = list(bm)
        for i, (c, lab) in enumerate(((BLUE, "top 10"), (RED, "top 5"),
                                      (GREEN, "top 1"))):
            key = [2, 1, 0][i]
            ax.plot(xs, [bm[x][key] for x in xs], "o-", color=c, lw=1.6,
                    ms=5, label=lab)
        # Best published tensor-network results, for scale.
        for y, txt in ((5.56, "best tensor network, top 10"),
                       (1.26, "best tensor network, top 1")):
            ax.axhline(y, color=GREY, lw=0.9, ls=":")
            ax.text(xs[0], y + 0.4, txt, fontsize=8, color=GREY)
        style(ax, "epoch", "correct caption found (%)",
              "Test recall, 5,000 images")
        ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "03_test.png", dpi=150); plt.close(fig)
        print(f"  benchmark points: {sorted(bm)}")

    # 4. what the model is learning ----------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(ep, [r["gamma"] for r in rows], color=BLUE, lw=1.6)
    style(ax[0], "epoch", "similarity gap",
          "True pairs versus the rest")

    ax[1].plot(ep, [r["hnm"] for r in rows], color=RED, lw=1.6)
    ax[1].axhline(0, color=GREY, lw=0.9, ls=":")
    style(ax[1], "epoch", "margin",
          "Margin over the closest wrong caption")

    g = [(r["epoch"], r["grad"]) for r in rows if r["grad"] is not None]
    if g:
        ax[2].plot([p[0] for p in g], [p[1] for p in g], color=GREEN, lw=1.6)
    style(ax[2], "epoch", "gradient norm", "Gradient norm")
    fig.tight_layout(); fig.savefig(OUT / "04_learning.png", dpi=150); plt.close(fig)

    # 5. how model size compares -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = ["quantum\n(this work)", "classical\ntensor network", "CLIP\ntext encoder"]
    params = [1_156_525, 2_310_327_176, 63_000_000]
    scores = [3.08, 1.72, 48.4]
    ax.scatter(params, scores, s=120, color=[BLUE, RED, GREY], zorder=3)
    for n, p, s in zip(names, params, scores):
        ax.annotate(n, (p, s), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)
    ax.set_xscale("log")
    style(ax, "text model parameters", "top-1 recall (%)",
          "Accuracy against model size")
    ax.set_ylim(0, 55)
    fig.tight_layout(); fig.savefig(OUT / "05_size.png", dpi=150); plt.close(fig)

    for f in sorted(OUT.glob("0*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
